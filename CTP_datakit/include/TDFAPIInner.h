#ifndef __TDF_API_INNER_H__
#define __TDF_API_INNER_H__
#pragma pack(push)
#pragma pack(1)

#include "TDFAPIStruct.h"
#include "TDCore.h"
#include "TDFDataDefine.h"

//��������
enum TDF_ENVIRON_SETTING_INNER
{
	TDF_ENVIRON_ORIGI_MODE = 50, //ʹ���ں˽ṹ�ص�
	TDF_ENVIRON_CANCEL_CODEEVENT = 51, //ȡ�������¼�
	TDF_ENVIRON_ALL_EVENT = 52, //��Ҫ���е��¼�֪ͨ��˫��Ҳ���������
};

enum TDF_MSG_ID_INNER
{
	MSG_DATA_ORIGINAL = 99,		//��Ӧ��Ϣ�ṹ��TDF_ORIDATA_MSG
	MSG_DATA_DFPacket = 100,	//��Ӧ��Ϣ�ṹ��TDF_DynDFPacket
};

enum ORI_MSG_DATA_TYPE
{
	ORI_SNAPSHOT = 0,//����(�������պ�ί�ж��ж��б仯)
	ORI_ORDERQUEUE = 1,//ί�ж��У��۹ɾ����̶���(��ί�ж��б仯)
};

//ԭʼ���ݽṹ
struct TDF_ORIDATA_MSG
{
	int						nCodeDate;			//���������
	char					marketKey[12];		//markeyKey,TW-1-0,TWO-1-0
	TDMarketData_Packet     marketData;         //ԭʼ���ݽṹ
	ORI_MSG_DATA_TYPE		eMsgDataType;		//ԭʼ�ṹ�������ͣ�����+ί�ж��л��ί�ж���
	int						nSide;				//ί�ж��з�����:0x01����:0x02����ͬʱ����
};

//���ݰ�������TDMarketData_Packet�Ͷ�Ӧ���͵����ݣ��ں�ģʽ��Ч
struct TDF_DynDFPacket
{
	char								marketKey[12];	//markeyKey,TW-1-0,TWO-1-0
	const TDFDynData_DataFeed_Packet*	pDFPacket;		//��Ϣ�ṹ
};

#pragma  pack(pop)
#endif