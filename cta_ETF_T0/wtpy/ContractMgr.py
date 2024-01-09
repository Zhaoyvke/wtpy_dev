import json
import yaml
import chardet

from .ProductMgr import ProductMgr, ProductInfo

class ContractInfo:

    def __init__(self):
        self.exchg:str = ''     #交易所
        self.code:str = ''      #合约代码
        self.name:str = ''      #合约名称
        self.product:str = ''   #品种代码
        self.stdCode:str = ''   #标准代码

        self.isOption:bool = False      # 是否期权合约
        self.underlying:str = ''        # underlying
        self.strikePrice:float = 0      # 行权价
        self.underlyingScale:float = 0  # 放大倍数


class ContractMgr:

    def __init__(self, prodMgr:ProductMgr = None):
        self.__contracts__ = dict()
        self.__underlyings__ = dict()   # 期权专用
        self.__products__ = dict()   # 期权专用
        self.__prod_mgr__ = prodMgr

    def load(self, fname:str):
        '''
        从文件加载品种信息
        '''
        f = open(fname, 'rb')
        content = f.read()
        f.close()
        encoding = chardet.detect(content[:500])["encoding"]
        content = content.decode(encoding)

        if fname.lower().endswith(".yaml"):
            exchgMap = yaml.full_load(content)
        else:
            exchgMap = json.loads(content)
        for exchg in exchgMap:
            exchgObj = exchgMap[exchg]

            for code in exchgObj:
                cObj = exchgObj[code]
                cInfo = ContractInfo()
                cInfo.exchg = exchg
                cInfo.code = code
                cInfo.name = cObj["name"]
                if 'volumescale' in cObj:
                    cInfo.multiplier = cObj["volumescale"]
                
                if 'expiredate' in cObj:
                    cInfo.maturity = cObj['expiredate']

                if "product" in cObj:
                    cInfo.product = cObj["product"]                    
                    #股票标准代码为SSE.000001，期货标准代码为SHFE.rb.2010
                    if cInfo.code[:len(cInfo.product)] == cInfo.product:
                        cInfo.stdCode = exchg + "." + cInfo.product + "." + cInfo.code[len(cInfo.product):]
                    else:
                        cInfo.stdCode = exchg + "." + cInfo.code

                    stdPID = exchg + "." + cInfo.product
                    if stdPID not in self.__products__:
                        self.__products__[stdPID] = list()

                    self.__products__[stdPID].append(cInfo.stdCode)
                else:
                    cInfo.product = cInfo.code
                    cInfo.stdCode = exchg + "." + cInfo.code
                    if "rules" in cObj:
                        pObj = cObj["rules"]
                        pInfo = ProductInfo()
                        pInfo.exchg = exchg
                        pInfo.product = cInfo.code
                        pInfo.name = cInfo.name
                        pInfo.session = pObj["session"]
                        pInfo.volscale = int(pObj["volscale"])
                        pInfo.pricetick = float(pObj["pricetick"])

                        if "minlots" in pObj:
                            pInfo.minlots = float(pObj["minlots"])
                        if "lotstick" in pObj:
                            pInfo.lotstick = float(pObj["lotstick"])

                if "option" in cObj:
                    oObj = cObj["option"]
                    cInfo.isOption = True
                    cInfo.underlying = oObj["underlying"]
                    cInfo.strikePrice = oObj["strikeprice"] #luke
                    cInfo.underlyingScale = float(oObj["underlyingscale"])
                    cInfo.type = oObj['optiontype']
                    if 'adjusted' in oObj:
                        cInfo.adjusted = oObj["adjusted"] #luke
                        cInfo.adjDate = oObj["adjdate"] #luke
                    if 'volumescale' in oObj:
                        cInfo.volumescale = oObj["volumescale"] #luke
                    if 'multiplier' in oObj:
                        cInfo.multiplier = oObj["multiplier"] #luke

                key = "%s.%s" % (exchg, code)
                self.__contracts__[key] = cInfo
                if cInfo.isOption:
                    stdUnderlying = f"{exchg}.{cInfo.underlying}"
                    if stdUnderlying not in self.__underlyings__:
                        self.__underlyings__[stdUnderlying] = list()

                    self.__underlyings__[stdUnderlying].append(cInfo.stdCode)

    def getContractInfo(self, stdCode:str) -> ContractInfo:
        '''
        获取合约信息
        @stdCode    合约代码，格式如SHFE.rb.2305
        '''
        if stdCode[-1] == '+' or stdCode[-1] == '-':
            stdCode = stdCode[:-1]
        else:
            items = stdCode.split(".")
            if len(items) == 3:
                stdCode = items[0] + "." + items[1] + items[2]
        if stdCode not in self.__contracts__:
            return None
            
        return self.__contracts__[stdCode]

    def getTotalCodes(self) -> list:
        '''
        获取全部合约代码列表
        '''
        codes = list()
        for code in self.__contracts__:
            codes.append(self.__contracts__[code].stdCode)
        return codes
    
    def getCodesByUnderlying(self, underlying:str) -> list:
        '''
        根据underlying读取合约列表
        @underlying 格式如CFFEX.IM2304
        '''
        if underlying in self.__underlyings__:
            return self.__underlyings__[underlying]
        return []
    
    def getCodesByProduct(self, stdPID:str) -> list:
        '''
        根据品种代码读取合约列表
        @stdPID 品种代码，格式如SHFE.rb
        '''
        if stdPID in self.__products__:
            return self.__products__[stdPID]
        return []
        

