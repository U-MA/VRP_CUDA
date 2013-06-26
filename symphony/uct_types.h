#ifndef _UCT_TYPES_H
#define _UCT_TYPES_H

//! 探索木のノードの子を表す構造体(random simulation用)
/*!
 * TODO この構造体の詳細説明
 */
typedef struct child {
	int customer; //!< 親のvehicleで訪れる顧客の番号
	int count; //!< このchildを訪れた回数
	double distance_ave; //!< このchildからシミュレーションを行った結果全体の平均
	int next; //!< このchildを表す探索木上での番号。node配列のインデックス値
} CHILD;

//! 探索木のノードを表す構造体(random simulation用)
/*! 
 * TODO この構造体の詳細説明
 */
typedef struct node {
	int vehicle; //!< 走行している車体の番号
	int idx; //!< メンバvehicleが訪れた顧客の数
	int nc; //!< 子の数(the Number of Child)
	CHILD *child;//[CHILD_MAX]; //<! 子の配列
	int sum_cnt; //!< 子のメンバcountの合計 
} NODE;


//! 探索木のノードの子を表す構造体(cws method用)
typedef struct cws_child {
	int savings_idx;
	int count;
	double distance_ave;
	int next;
} CWS_CHILD;

//! 探索木のノードを表す構造体(cws method用)
typedef struct cws_node {
	int nc;
	CWS_CHILD *child;
	int sum_cnt;
} CWS_NODE;

//! Route DATA; ルートの情報を管理する構造体
typedef struct rdata {
	int num_customers; //! 倉庫を含む顧客の数
	int num_routes;    //! ルートの数
	int capacity;      //! ルートの容量
	int *route;        //! ルート配列, 二次元配列として扱う
	int *route_size;   //! 各ルートに含まれている顧客の数(倉庫は含まない)
	int *route_cap;    //! 各ルートが訪問した顧客のdemandsの和
} RDATA;
#endif
