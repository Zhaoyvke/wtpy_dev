#ifndef __TDF_API_INNER_H__
#define __TDF_API_INNER_H__
#pragma pack(push)
#pragma pack(1)

#include "TDFAPIStruct.h"
#include "TDCore.h"
#include "TDFDataDefine.h"

//环境设置
enum TDF_ENVIRON_SETTING_INNER
{
	TDF_ENVIRON_ORIGI_MODE = 50, //使用内核结构回调
	TDF_ENVIRON_CANCEL_CODEEVENT = 51, //取消代码事件
	TDF_ENVIRON_ALL_EVENT = 52, //需要所有的事件通知，双活也不允许过滤
};

enum TDF_MSG_ID_INNER
{
	MSG_DATA_ORIGINAL = 99,		//对应消息结构体TDF_ORIDATA_MSG
	MSG_DATA_DFPacket = 100,	//对应消息结构体TDF_DynDFPacket
};

enum ORI_MSG_DATA_TYPE
{
	ORI_SNAPSHOT = 0,//快照(基础快照和委托队列都有变化)
	ORI_ORDERQUEUE = 1,//委托队列，港股经纪商队列(仅委托队列变化)
};

//原始数据结构
struct TDF_ORIDATA_MSG
{
	int						nCodeDate;			//代码表日期
	char					marketKey[12];		//markeyKey,TW-1-0,TWO-1-0
	TDMarketData_Packet     marketData;         //原始数据结构
	ORI_MSG_DATA_TYPE		eMsgDataType;		//原始结构数据类型，快照+委托队列或仅委托队列
	int						nSide;				//委托队列方向，卖:0x01，买:0x02，可同时存在
};

//数据包，包含TDMarketData_Packet和对应类型的数据，内核模式有效
struct TDF_DynDFPacket
{
	char								marketKey[12];	//markeyKey,TW-1-0,TWO-1-0
	const TDFDynData_DataFeed_Packet*	pDFPacket;		//消息结构
};

#pragma  pack(pop)
#endif