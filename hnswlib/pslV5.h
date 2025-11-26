#ifndef DIS_H_
#define DIS_H_

// V2: judge with small 

#include<omp.h>
#include<string>
#include<vector>
#include<cstring>
#include<algorithm>
#include<cstdio>
#include <xmmintrin.h>
#include <stdint.h>
#include <unordered_set>

#include "bloomfilter.h"

using namespace std;
#define MAXN 1000000000
//#define MAXT 2000000000
#define MAXT 65000
#define MAXD 120
#define MAXINT ((unsigned) 4294967295)
#define LOAD_INDEP_LABEL false
#define COMPUTE_INDEP_LABEL false
//#define MAXDIS 64
#define N_ROOTS 16
#define MAX_BP_THREADS 8
#define N_OMP_THREADS 8

#define MAXLINE 1024

#define RANK_STATIC 0
#define RANK_DYNAMIC_DEC 1
#define RANK_DYNAMIC_INC 2

#define GEN_MAXIMAL_INDEP false


// hash related - KMV
#define N_HASHFUNC 16
#define POS_ENHASH 4
#define HASH_INT_LIM 1.0
#define HASH_SEED 93

typedef unsigned short tint;
//typedef int tint;
typedef long long lint;

using namespace BloomFilter;

// namespace hnswlib{
class SV {
public:
	SV();
	SV(int id, long long val);
	int id;
	long long val;
	bool operator < (const SV& v) const;
};

class LinkNode {
public:
	LinkNode *prev;
	LinkNode *next;
	int id; int val;
public:
	LinkNode();
};

class LinearHeap {
public:
	int n, maxrank;
	LinkNode *nods;
	LinkNode *rank;

public:
	LinearHeap(int n, int maxrank);
	void link(int v, int r);
	void unlink(int v);
	bool is_linked(int v);
	int get_rank(int v);
	bool has_rank(int r);
	int get_first_in_rank(int r);
	~LinearHeap();
};

class BFLabelHashEle {
	public:
	int n_ele;
	
	unsigned int maxmov;
	BloomFilter::bloom_filter bf;
	static constexpr double false_positive_probability = 0.05;
	// static constexpr double projected_element_divisor = 1.0;

	BFLabelHashEle(): n_ele(0){};

	BFLabelHashEle(std::vector<unsigned int> & label, int start, int n_ele_, int max_size, unsigned int maxmov_): n_ele(n_ele_), maxmov(maxmov_) {
		BloomFilter::bloom_parameters params;
		// params.projected_element_count = static_cast<unsigned long long>(max_size / projected_element_divisor);
		params.projected_element_count = max_size;
		params.random_seed = HASH_SEED;
		// params.false_positive_probability = 0.01;
		params.false_positive_probability = false_positive_probability;
		params.compute_optimal_parameters();
		bf = BloomFilter::bloom_filter(params);
		// std::cout << "nele:" << n_ele_ << "maxsize" << max_size<< std::endl;
		for (int i=0; i<n_ele_; i++) bf.insert(label[start+i]>>maxmov);
		// if (bf.contains(pos[0])) std::cout << "test: true" << std::endl;
		// std::cout << bf.element_count()<< std::endl;
		// std::cout << bf.table_size_ << std::endl;
		// std::cout << bf.salt_.size() << std::endl;
	}

	bool isIntersect(std::vector<unsigned int>& label, int start, int size) {
		for (int i=0; i<size; i++){
			if (bf.contains(label[start+i]>>maxmov)) return true;
		}
		return false;
	}
	bool isIntersect(BFLabelHashEle & bf2) {
		BloomFilter::bloom_filter bft = bf | bf2.bf;
		
		return (bf.estCard()+bf2.bf.estCard() -bft.estCard()) >= HASH_INT_LIM;
	}
};

struct BPLabel {
    uint8_t bpspt_d[N_ROOTS];
    uint64_t bpspt_s[N_ROOTS][2];
};

class DisOracle{
public:
	int n, *nid, nown;
	unsigned MAXDIS, MAXMOV, MASK;
	vector<unsigned> *label;
	bool* is_indep;
	long long m;
	int ** pos;
	BFLabelHashEle ** hashes;

	DisOracle();
	DisOracle(std::vector<std::pair<int,int>> &el, int max_range, bool consider_indep);
	void load_graph(string path);
	void construct_bp_label(int max_range);
	void save_idx(string path);
	void load_idx(string path);
	inline int query(int u, int v);
	inline int query_by_nid(int u, int v);
	inline int query_by_bp(int u, int v);
	inline bool prune_by_bp(int u, int v, int d);
	inline bool can_update(int v, int dis, char *nowdis);
	inline int query_by_t(int u, int v);
	~DisOracle();

public:
	int **con, *dat, *deg;
	tint *last_t;
	char *dis;
	tint t;
	bool *usd_bp;
	BPLabel *label_bp;

	void init_query();

public:
	static inline bool get_edge(char *line, int &a, int &b, int num_cnt = 2);
	static inline int get_num_cnt(string path);
	static void get_order(	vector<int> *con, int n, int *o);
	static void create_bin(string path, int merge_equv = 0, int rank_method = RANK_STATIC);
	void create_ord(vector<pair<int, int>> &);
	void see_labels();
	bool query_by_bp_es(int u, int v, int search_k);
	bool query_by_nid_es(int u, int v, int search_k);
	bool query_by_t_es(int u, int v, int search_k);
	bool query(int u, int v, int search_k);
	void const_hash();
	void see_hash();

	void init_query_node(int q);
};

//======implementation========


bool SV::operator < (const SV& v) const {
	if( val > v.val ) return true;
	if( val < v.val ) return false;
	return id < v.id;
}

SV::SV(int id, long long val) {
	this->id = id; this->val = val;
}

SV::SV() {id = -1; val=-1;}


LinkNode::LinkNode() {
	prev = NULL;
	next = NULL;
	id = -1;
	val = -1;
}

LinearHeap::LinearHeap(int n, int maxrank) {
	this->n = n;
	this->maxrank = maxrank;
	nods = new LinkNode[n];
	for( int i = 0; i < n; ++i) nods[i].id = i;
	rank = new LinkNode[maxrank+1];
}

LinearHeap::~LinearHeap() {
	delete[] nods;
	delete[] rank;
}

void LinearHeap::unlink(int v) {
	LinkNode *nod = nods + v;
	if(nod->prev) nod->prev->next = nod->next;
	if(nod->next) nod->next->prev = nod->prev;
	nod->prev = NULL;
	nod->next = NULL;
}

void LinearHeap::link(int v, int r) {
	LinkNode *nod = nods + v;
	nod->val = r;
	if(nod->prev) nod->prev->next = nod->next;
	if(nod->next) nod->next->prev = nod->prev;
	nod->prev = rank+r;
	nod->next = rank[r].next;
	if(rank[r].next) rank[r].next->prev = nod;
	rank[r].next = nod;
}

bool LinearHeap::is_linked(int v) {
	return nods[v].prev != NULL;
}

int LinearHeap::get_rank(int v) {
	return nods[v].val;
}

bool LinearHeap::has_rank(int r) {
	return rank[r].next != NULL;
}

int LinearHeap::get_first_in_rank(int r) {
	return rank[r].next == NULL ? -1 : rank[r].next->id;
}

bool DisOracle::get_edge(char *line, int &a, int &b, int num_cnt) {
	if( !isdigit(line[0]) ) return false;
	vector<char*> v_num;
	int len = (int) strlen(line);
	for( int i = 0; i < len; ++i )
		if( !isdigit(line[i]) && line[i] != '.') line[i] = '\0';
		else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
	if( (int) v_num.size() != num_cnt ) return false;
	sscanf( v_num[0], "%d", &a );
	sscanf( v_num[1], "%d", &b );
	return true;
}

int DisOracle::get_num_cnt(string path) {
	FILE *fin = fopen( (path + "graph.txt").c_str(), "r" );
	char line[MAXLINE], st[64];
	int cnt = 0, min_cnt = 100;

	while( fgets( line, MAXLINE, fin ) && cnt < 10 ) {
		if( !isdigit(line[0]) ) continue;
		vector<char*> v_num;
		int len = (int) strlen(line);
		for( int i = 0; i < len; ++i )
			if( !isdigit(line[i]) && line[i] != '.' ) line[i] = '\0';
			else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
		if( (int) v_num.size() < 2 ) continue;
		min_cnt = min(min_cnt, (int) v_num.size());
		++cnt;
	}
	fclose( fin );
	return min_cnt;
}

void DisOracle::get_order(	vector<int> *con, int n, int *o) {
    printf( "Ranking Method = RANK_STATIC\n" );
    SV *f = new SV[n];
    for( int i = 0; i < n; ++i )
        f[i].id = i, f[i].val = (long long) con[i].size();
    sort(f, f + n);
    for(int i = 0; i < n; ++i) o[i] = f[i].id;
    delete[] f;
    
    // not used
	if(GEN_MAXIMAL_INDEP) {
		printf( "Maximal Independent Set:\n" );
		bool* choose = new bool[n];
		vector<int> v_ind;

		memset( choose, 0, sizeof(bool) * n);
		for( int i = n-1; i >= 0; --i ) {
			bool flag = true;
			int v = o[i];
			for( int j = 0; j < (int) con[v].size(); ++j )
				if(choose[con[v][j]]) {
					flag = false;
					break;
				}
			if( flag ) {
				choose[v] = true;
				v_ind.push_back(v);
			}
		}
		int nown = 0;
		for( int i = 0; i < n; ++i )
			if( !choose[o[i]] ) o[nown++] = o[i];
		for( int i = v_ind.size() - 1; i >= 0; --i )
			o[nown++] = v_ind[i];
		delete[] choose;
		printf( "nown=%d, n=%d.\n", nown, n );
	}
}

void DisOracle::create_ord(vector<pair<int, int>> &el) {
    int merge_equv = false; // disable merge equivalence
    // string path, int merge_equv, int rank_method ) {
	// FILE *fin = fopen( (path + "graph.txt").c_str(), "r" );
	// char line[MAXLINE], st[64];
	n = 0;
	int a, b, num_cnt;
	// vector< pair<int,int> > el;
	lint cnt = 0;
	m = 0;
    for (auto pair: el) {
        n = max(max(n, pair.first+1), pair.second+1);
        cnt ++;
    }
	// printf( "Loading text, num_cnt=%d...\n", num_cnt );
	// while( fgets( line, MAXLINE, fin ) ) {
	// 	if( !get_edge(line, a, b, num_cnt) ) continue;
	// 	if( a < 0 || b < 0 || a == b ) continue;
	// 	el.push_back(make_pair(a, b));
	// 	n = max(max(n, a+1), b+1);
	// 	if( (++cnt) % (lint) 10000000 == 0 ) printf( "%lld lines finished\n", cnt );
	// }
	// fclose( fin );

	vector<int> *contmp = new vector<int>[n];
	printf( "Deduplicating...\n" );

	for(long long i = 0; i < el.size(); ++i) {
		contmp[el[i].first].push_back(el[i].second);
		contmp[el[i].second].push_back(el[i].first);
	}

	for( int i = 0; i < n; ++i )
		if( contmp[i].size() > 0 ){
			sort( contmp[i].begin(), contmp[i].end() );
			int p = 1;
			for( int j = 1; j < (int) contmp[i].size(); ++j )
				if( contmp[i][j-1] != contmp[i][j] ) contmp[i][p++] = contmp[i][j];
			contmp[i].resize( p ); m += p;
		}

	long long *f1 = new long long[n];
	memset( f1, 0, sizeof(long long) * n );

	long long *f2 = new long long[n];
	memset( f2, 0, sizeof(long long) * n );

	if( !merge_equv ) {
		for( int i = 0; i < n; ++i ) f1[i] = i, f2[i] = i;
	} else {
	printf( "Merging...\n" );
		long long s = 0;
		long long *nows = new long long[m+n+1];
		int *nowt = new int[m+n+1];
		memset( nowt, 0, sizeof(int) * (m+n+1) );

		for( int v = 0; v < n; ++v )
			for( int i = 0; i < (int) contmp[v].size(); ++i ) {
				int u = contmp[v][i];
				if( nowt[f1[u]] != (v+1) ) {
					++s;
					nows[f1[u]] = s;
					nowt[f1[u]] = (v+1);
					f1[u] = s;
				} else f1[u] = nows[f1[u]];
			}

		for( int v = 0; v < n; ++v )
			if( nowt[f1[v]] != -1 ) {
				nows[f1[v]] = v;
				nowt[f1[v]] = -1;
				f1[v] = v;
			} else f1[v] = nows[f1[v]];

		s = 0;
		memset( nowt, 0, sizeof(int) * (m+n+1) );
		for( int v = 0; v < n; ++v )
			for( int i = 0; i <= (int) contmp[v].size(); ++i ) {
				int u = (i == (int) contmp[v].size()) ? v : contmp[v][i];
				if( nowt[f2[u]] != (v+1) ) {
					++s;
					nows[f2[u]] = s;
					nowt[f2[u]] = (v+1);
					f2[u] = s;
				} else f2[u] = nows[f2[u]];
			}

		for( int v = 0; v < n; ++v )
			if( nowt[f2[v]] != -1 ) {
				nows[f2[v]] = v;
				nowt[f2[v]] = -1;
				f2[v] = v;
			} else f2[v] = nows[f2[v]];

		delete[] nows; delete[] nowt;

		int cnt1_n = 0, cnt1_m = 0, cnt2_n = 0, cnt2_m = 0;
		for( int i = 0; i < n; ++i ) {
			if( f1[i] != i ) {
				++cnt1_n;
				cnt1_m += (int) contmp[i].size();
			}
			if( f2[i] != i ) {
				++cnt2_n;
				cnt2_m += (int) contmp[i].size();
			}
		}

		m = 0;
		for( int i = 0; i < n; ++i ) {
			if( f1[i] != i || f2[i] != i ) {contmp[i].clear(); continue;}
			int p = 0;
			for( int j = 0; j < (int) contmp[i].size(); ++j ) {
				int v = contmp[i][j];
				if( f1[v] == v && f2[v] == v ) contmp[i][p++] = v;
			}
			contmp[i].resize(p); m += p;
		}
		printf( "cnt1_n = %d, cnt1_m = %d, cnt2_n = %d, cnt2_m = %d, m = %lld\n", cnt1_n, cnt1_m, cnt2_n, cnt2_m, m );
	}
	printf( "Reordering...\n" );
	int *f = new int[n];
	get_order(contmp, n, f);

	int *oid = new int[n];
	nid = new int[n];
	for( int i = 0; i < n; ++i )
		oid[i] = f[i], nid[f[i]] = i;

	for( int i = 0; i < n; ++i ) {
		if(f1[i] != i) nid[i] = -nid[f1[i]]-1;
		if(f2[i] != i) nid[i] = MAXN + nid[f2[i]];
	}

	//for(int i = 0; i < SCOREDIS; ++i) delete[] score[i];
	//delete[] score;

	printf( "Creating adjacency list...\n" );
	dat = new int[m];
	deg = new int[n];
	int **adj = new int *[n];

	lint pos = 0;
	for( int i = 0; i < n; ++i ) {
		adj[i] = dat + pos;
		pos += (int) contmp[oid[i]].size();
	}
	memset( deg, 0, sizeof(int) * n );

	for( int i = 0; i < n; ++i ) {
		int ii = oid[i];
		for( int p = 0; p < (int) contmp[ii].size(); ++p ) {
			int jj = contmp[ii][p];
			int j = nid[jj];
			adj[j][deg[j]++] = i;
		}
	}

	con = new int*[n];
	long long p = 0;
	for (int i=0; i<n; ++i) {con[i] = dat+p; p+= deg[i];}

	delete[] adj; delete[] f; delete[] oid; 
	delete[] f1; delete[] f2;
}

void DisOracle::const_hash() {
	hashes = new BFLabelHashEle*[n];
	int maxcnt = -1;
	for (int i=0; i<n; i++) {
		for (int j=POS_ENHASH; j<=MAXDIS; j++) {
			// int size = j==0?pos[i][0]:
			int size = j==0?pos[i][0]:pos[i][j]-pos[i][j-1];
			maxcnt = max(maxcnt, size);

		}
	}
	std::cout << "size:"<< maxcnt << std::endl;
	for (int i=0; i<n; i++) hashes[i] = new BFLabelHashEle[MAXDIS-POS_ENHASH+1];
	for (int i=0; i<n; i++) {
		for (int j=POS_ENHASH; j<=MAXDIS; j++) {
			// std::cout << "const:" << i << std::endl;
			int loc = j==0?0:pos[i][j-1];
			int size = pos[i][j]-loc;
			// std::cout << "i: " << i << "size:" << size << std::endl;
			hashes[i][j-POS_ENHASH] = BFLabelHashEle(label[i], loc, size, maxcnt, MAXMOV);
		}
	}
	// std::cout << "table size: " << hashes[49998][MAXDIS-POS_ENHASH].bf.table_size_ << std::endl;
}

void DisOracle::see_hash() {
	BloomFilter::bloom_filter bf = hashes[0][0].bf;
	std::cout << "hash func: " << bf.salt_.size() << std::endl;
	for (int i=0; i<bf.salt_.size(); i++) std::cout << bf.salt_[i] << std::endl;
	bf = hashes[1][0].bf;
	std::cout << "hash func: " << bf.salt_.size() << std::endl;
	for (int i=0; i<bf.salt_.size(); i++) std::cout << bf.salt_[i] << std::endl;
}


// void DisOracle::create_bin(string path, int merge_equv, int rank_method ) {
// 	FILE *fin = fopen( (path + "graph.txt").c_str(), "r" );
// 	char line[MAXLINE], st[64];
// 	int n = 0, a, b, num_cnt = get_num_cnt(path);
// 	vector< pair<int,int> > el;
// 	lint cnt = 0, m = 0;

// 	printf( "Loading text, num_cnt=%d...\n", num_cnt );
// 	while( fgets( line, MAXLINE, fin ) ) {
// 		if( !get_edge(line, a, b, num_cnt) ) continue;
// 		if( a < 0 || b < 0 || a == b ) continue;
// 		el.push_back(make_pair(a, b));
// 		n = max(max(n, a+1), b+1);
// 		if( (++cnt) % (lint) 10000000 == 0 ) printf( "%lld lines finished\n", cnt );
// 	}
// 	fclose( fin );

// 	vector<int> *con = new vector<int>[n];
// 	printf( "Deduplicating...\n" );

// 	for(long long i = 0; i < el.size(); ++i) {
// 		con[el[i].first].push_back(el[i].second);
// 		con[el[i].second].push_back(el[i].first);
// 	}

// 	for( int i = 0; i < n; ++i )
// 		if( con[i].size() > 0 ){
// 			sort( con[i].begin(), con[i].end() );
// 			int p = 1;
// 			for( int j = 1; j < (int) con[i].size(); ++j )
// 				if( con[i][j-1] != con[i][j] ) con[i][p++] = con[i][j];
// 			con[i].resize( p ); m += p;
// 		}

// 	long long *f1 = new long long[n];
// 	memset( f1, 0, sizeof(long long) * n );

// 	long long *f2 = new long long[n];
// 	memset( f2, 0, sizeof(long long) * n );

// 	if( !merge_equv ) {
// 		for( int i = 0; i < n; ++i ) f1[i] = i, f2[i] = i;
// 	} else {
// 	printf( "Merging...\n" );
// 		long long s = 0;
// 		long long *nows = new long long[m+n+1];
// 		int *nowt = new int[m+n+1];
// 		memset( nowt, 0, sizeof(int) * (m+n+1) );

// 		for( int v = 0; v < n; ++v )
// 			for( int i = 0; i < (int) con[v].size(); ++i ) {
// 				int u = con[v][i];
// 				if( nowt[f1[u]] != (v+1) ) {
// 					++s;
// 					nows[f1[u]] = s;
// 					nowt[f1[u]] = (v+1);
// 					f1[u] = s;
// 				} else f1[u] = nows[f1[u]];
// 			}

// 		for( int v = 0; v < n; ++v )
// 			if( nowt[f1[v]] != -1 ) {
// 				nows[f1[v]] = v;
// 				nowt[f1[v]] = -1;
// 				f1[v] = v;
// 			} else f1[v] = nows[f1[v]];

// 		s = 0;
// 		memset( nowt, 0, sizeof(int) * (m+n+1) );
// 		for( int v = 0; v < n; ++v )
// 			for( int i = 0; i <= (int) con[v].size(); ++i ) {
// 				int u = (i == (int) con[v].size()) ? v : con[v][i];
// 				if( nowt[f2[u]] != (v+1) ) {
// 					++s;
// 					nows[f2[u]] = s;
// 					nowt[f2[u]] = (v+1);
// 					f2[u] = s;
// 				} else f2[u] = nows[f2[u]];
// 			}

// 		for( int v = 0; v < n; ++v )
// 			if( nowt[f2[v]] != -1 ) {
// 				nows[f2[v]] = v;
// 				nowt[f2[v]] = -1;
// 				f2[v] = v;
// 			} else f2[v] = nows[f2[v]];

// 		delete[] nows; delete[] nowt;

// 		int cnt1_n = 0, cnt1_m = 0, cnt2_n = 0, cnt2_m = 0;
// 		for( int i = 0; i < n; ++i ) {
// 			if( f1[i] != i ) {
// 				++cnt1_n;
// 				cnt1_m += (int) con[i].size();
// 			}
// 			if( f2[i] != i ) {
// 				++cnt2_n;
// 				cnt2_m += (int) con[i].size();
// 			}
// 		}

// 		m = 0;
// 		for( int i = 0; i < n; ++i ) {
// 			if( f1[i] != i || f2[i] != i ) {con[i].clear(); continue;}
// 			int p = 0;
// 			for( int j = 0; j < (int) con[i].size(); ++j ) {
// 				int v = con[i][j];
// 				if( f1[v] == v && f2[v] == v ) con[i][p++] = v;
// 			}
// 			con[i].resize(p); m += p;
// 		}
// 		printf( "cnt1_n = %d, cnt1_m = %d, cnt2_n = %d, cnt2_m = %d, m = %lld\n", cnt1_n, cnt1_m, cnt2_n, cnt2_m, m );
// 	}
// 	printf( "Reordering...\n" );
// 	int *f = new int[n];
// 	get_order(con, n, f, rank_method);

// 	int *oid = new int[n], *nid = new int[n];
// 	for( int i = 0; i < n; ++i )
// 		oid[i] = f[i], 
    
//     [f[i]] = i;

// 	for( int i = 0; i < n; ++i ) {
// 		if(f1[i] != i) nid[i] = -nid[f1[i]]-1;
// 		if(f2[i] != i) nid[i] = MAXN + nid[f2[i]];
// 	}

// 	//for(int i = 0; i < SCOREDIS; ++i) delete[] score[i];
// 	//delete[] score;

// 	printf( "Creating adjacency list...\n" );
// 	int *dat = new int[m], *deg = new int[n], **adj = new int *[n];

// 	lint pos = 0;
// 	for( int i = 0; i < n; ++i ) {
// 		adj[i] = dat + pos;
// 		pos += (int) con[oid[i]].size();
// 	}
// 	memset( deg, 0, sizeof(int) * n );

// 	for( int i = 0; i < n; ++i ) {
// 		int ii = oid[i];
// 		for( int p = 0; p < (int) con[ii].size(); ++p ) {
// 			int jj = con[ii][p];
// 			int j = nid[jj];
// 			adj[j][deg[j]++] = i;
// 		}
// 	}

// 	printf( "Saving binary...\n" );
// 	FILE *fout = fopen( (path + "graph-dis.bin").c_str(), "wb" );

// 	fwrite( &n, sizeof(int), 1, fout );
// 	fwrite( &m, sizeof(lint), 1, fout );
// 	fwrite( deg, sizeof(int), n, fout );
// 	fwrite( dat, sizeof(int), m, fout );
// 	fwrite( nid, sizeof(int), n, fout );
// 	fclose( fout );

// 	printf( "Created binary file, n = %d, m = %lld\n", n, m );
// 	delete[] adj; delete[] deg; delete[] dat; delete[] f; delete[] con; delete[] oid; delete[] nid;
// 	delete[] f1; delete[] f2;
// }



bool DisOracle::can_update(int v, int dis, char *nowdis) {
	for( int i = 0; i < (int) label[v].size(); ++i ) {
		int w = label[v][i]>>MAXMOV, d = label[v][i]&MASK;
		if( nowdis[w] >= 0 && nowdis[w] + d <= dis ) return false;
	}
	return true;
}

DisOracle::DisOracle() {
	MAXDIS = 0; MAXMOV = 0; MASK = 0;
	n = 0; nid = NULL; label = NULL; is_indep = NULL; m = 0; nown = 0;
	last_t = NULL; dis = NULL; t = 0; con = NULL; dat = NULL; deg = NULL;
	label_bp = NULL; usd_bp = NULL;
}

DisOracle::DisOracle(std::vector<std::pair<int,int>> &el, int max_range, bool consider_indep) {
	// we do not load graph, but directly use the data in create_bin
	// load_graph(path);
	omp_set_num_threads(N_OMP_THREADS);

	// max_range -= 1;
	create_ord(el);
	nown = n;
	construct_bp_label(max_range+1);

	MAXDIS = 2; MAXMOV = 1;
	std::cout << "nown" << nown << std::endl;
	while( MAXINT / (nown * 2) >= MAXDIS && MAXDIS<=max_range) {MAXDIS *= 2;++MAXMOV;}
	MASK = MAXDIS - 1;

	MAXDIS = MAXDIS<max_range?MAXDIS:max_range;

	printf( "MAXDIS=%d, nown=%d, consider_indep=%s\n", MAXDIS, nown, consider_indep?"true":"false" );

	pos = new int*[n];
	for( int i = 0; i < n; ++i ) pos[i] = new int[MAXDIS];
	label = new vector<unsigned>[n];
	is_indep = new bool[n];
	memset(is_indep, 0, sizeof(bool) * n);

	for(int i = 0; i < n; ++i) {
		if( consider_indep ) {
			bool flag = true;
			for( int j = 0; j < deg[i]; ++j )
				if( con[i][j] > i ) { flag = false; break; }
			if( flag ) is_indep[i] = true;
		}
		if( !is_indep[i] ) {
			if( !usd_bp[i] ) {
				label[i].push_back((((unsigned)i)<<MAXMOV) | 0);
				for( int j = 0; j < deg[i] && con[i][j] < i; ++j )
					if( !usd_bp[con[i][j]] )
						label[i].push_back((((unsigned) con[i][j]) << MAXMOV) | 1);
				pos[i][0] = 1; pos[i][1] = (int) label[i].size();
			} else {pos[i][0] = 0; pos[i][1] = 0;}
		}
	}

	int dis = 2;
	for( long long cnt = 1, cnt_tmp = 0, cnt_red = 0, pre_cnt = 0; cnt && dis <= MAXDIS; ++dis ) {
		cnt = 0, cnt_tmp = 0, cnt_red = 0;
		vector<unsigned> *label_new = new vector<unsigned>[n];
		#pragma omp parallel
		{
			int pid = omp_get_thread_num(), np = omp_get_num_threads();
			long long local_cnt = 0, local_cnt_tmp = 0, local_cnt_red = 0;
			unsigned char *used = new unsigned char[n/8+1];
			memset( used, 0, sizeof(unsigned char) * (n/8+1) );
			vector<int> cand;
			char *nowdis = new char[n];
			memset( nowdis, -1, sizeof(char) * n );

			for( int u = pid; u < n; u += np ) {

				if(is_indep[u] || usd_bp[u]) continue;
				cand.clear();
				for( int i = 0; i < deg[u]; ++i ) {
					int w = con[u][i];
					if( !consider_indep || !is_indep[w] ) {
						for( int j = pos[w][dis-2]; j < pos[w][dis-1]; ++j ) {
							int v = label[w][j] >> MAXMOV;
							if( v >= u ) break;
							if( !(used[v/8]&(1<<(v%8))) ) used[v/8] |= (1<<(v%8)), cand.push_back(v); else ++local_cnt_red;
						}
					} else if (dis == 2) {
						for( int j = 0; j < deg[w]; ++j) {
							int v = con[w][j];
							if( v >= u ) break;
							if( !(used[v/8]&(1<<(v%8))) ) used[v/8] |= (1<<(v%8)), cand.push_back(v); else ++local_cnt_red;
						}
					} else if (dis > 2) {
						for( int k = 0; k < deg[w]; ++k ) {
							int x = con[w][k];
							if( x == u ) continue;
							for( int j = pos[x][dis-3]; j < pos[x][dis-2]; ++j ) {
								int v = label[x][j] >> MAXMOV;
								if( v >= u ) break;
								if( !(used[v/8]&(1<<(v%8))) ) used[v/8] |= (1<<(v%8)), cand.push_back(v); else ++local_cnt_red;
							}
						}
					}
				}

				int n_cand = 0;
				for( int i = 0; i < (int) label[u].size(); ++i ) nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;
				for( int i = 0; i < (int) cand.size(); ++i ) {
					used[cand[i]/8] = 0;
					if( !prune_by_bp(u, cand[i], dis) )
						if( can_update(cand[i], dis, nowdis) ) {cand[n_cand++] = cand[i];} else {++local_cnt_tmp;}
				}

				cand.resize(n_cand); sort(cand.begin(), cand.end());
				for( int i = 0; i < (int) cand.size(); ++i ) {
					//if(label_new[u].size() == label_new[u].capacity() && label_new[u].size() > 100) label_new[u].reserve( label_new[u].size() * 1.2 );
					label_new[u].push_back((((unsigned)cand[i])<<MAXMOV) | (unsigned) dis), ++local_cnt;
				}
				//vector<unsigned>(label_new[u]).swap(label_new[u]);
				for( int i = 0; i < (int) label[u].size(); ++i ) nowdis[label[u][i]>>MAXMOV] = -1;
			}

			if(pid==0) printf( "num_thread=%d", np);
			#pragma omp critical
			{
				cnt += local_cnt; cnt_tmp += local_cnt_tmp; cnt_red += local_cnt_red;
			}
			delete[] used; delete[] nowdis;
		}
		
		#pragma omp parallel
		{
			int pid = omp_get_thread_num(), np = omp_get_num_threads();
			for( int u = pid; u < n; u += np ) {
				label[u].insert(label[u].end(), label_new[u].begin(), label_new[u].end());
				vector<unsigned>(label[u]).swap(label[u]);
				vector<unsigned>().swap(label_new[u]);
				pos[u][dis] = (int) label[u].size();
			}
		}

		delete[] label_new; pre_cnt = cnt;
	}

	long long tt = 0, s = 0;
	
	for( int i = 0; i < n; ++i ) {
		// tt += label[i].size() * 4, s = max(s, (long long) label[i].size());
		if (POS_ENHASH > MAXDIS)
			tt += label[i].size();
		else {
			tt += POS_ENHASH;
		}
		
		s = max(s, (long long) label[i].size());
	}

	std::cout << "tmp:" << (tt/(1024*1024)) << std::endl;

	if(consider_indep && COMPUTE_INDEP_LABEL) {
		printf( "Processing Independent Set...\n" );
		long long cnt = 0, cnt_tmp = 0, cnt_red = 0;
		
		#pragma omp parallel
		{
			int pid = omp_get_thread_num(), np = omp_get_num_threads();
			long long local_cnt = 0, local_cnt_tmp = 0, local_cnt_red = 0, local_tt = 0, local_s = 0;
			unsigned char *used = new unsigned char[n/8+1];
			memset( used, 0, sizeof(unsigned char) * (n/8+1) );
			vector<unsigned> *cand = new vector<unsigned>[dis];
			char *nowdis = new char[n];
			memset( nowdis, -1, sizeof(char) * n );

			for( int u = pid; u < n; u += np ) {
				if( !is_indep[u] ) continue;
				for( int i = 0; i < dis; ++i ) cand[i].clear();
				cand[0].push_back(u); nowdis[u] = 0;
				for( int i = 0; i < deg[u]; ++i ) {cand[1].push_back(con[u][i]); nowdis[con[u][i]] = 1;}
				for( int d = 2; d < dis; ++d ) {
					for( int i = 0; i < deg[u]; ++i ) {
						int w = con[u][i];
						for( int j = pos[w][d-2]; j < pos[w][d-1]; ++j ) {
							int v = label[w][j] >> MAXMOV;
							if( !(used[v/8]&(1<<(v%8))) ) used[v/8] |= (1<<(v%8)), cand[d].push_back(v); else ++local_cnt_red;
						}
					}
					int n_cand = 0;
					for( int i = 0; i < (int) cand[d].size(); ++i ) {
						used[cand[d][i]/8] = 0;
						if( !prune_by_bp(u, cand[d][i], d) )
							if( can_update(cand[d][i], d, nowdis) ) cand[d][n_cand++] = cand[d][i]; else ++local_cnt_tmp;
					}
					cand[d].resize(n_cand); sort(cand[d].begin(), cand[d].end());
					for( int i = 0; i < (int) cand[d].size(); ++i ) nowdis[cand[d][i]] = d;
				}

				int n_cand = 0;
				for( int d = 0; d < dis; ++d ) {
					for( int i = 0; i < (int) cand[d].size(); ++i )
						nowdis[cand[d][i]] = -1;
					n_cand += (int) cand[d].size();

					if(LOAD_INDEP_LABEL)
						for( int i = 0; i < (int) cand[d].size(); ++i )
							label[u].push_back((((unsigned) cand[d][i])<<MAXMOV) | (unsigned) d);
				}
				if(LOAD_INDEP_LABEL) vector<unsigned>(label[u]).swap(label[u]);
				local_tt += n_cand * 4; local_s = max(local_s, (long long) n_cand); local_cnt += n_cand;
			}

			#pragma omp critical
			{
				cnt += local_cnt; cnt_tmp += local_cnt_tmp; cnt_red += local_cnt_red; tt += local_tt; s = max(s, local_s);
			}
			delete[] used; delete[] nowdis; delete[] cand;
		}

	}

	// printf( "sorting...\n");
	// {
	// 	int pid = 0, np = 1;
	// 	for( int u = pid; u < n; u += np ) sort( label[u].begin(), label[u].end() );
	// }

	const_hash();

	if (POS_ENHASH <= MAXDIS) {
		for (int i = 0; i < MAXDIS - POS_ENHASH + 1; i++) {
			// ull防止溢出
            unsigned long long hashsize = hashes[0][i].bf.memSize() + 4ULL;
			std::cout << "hash size for dis " << (i + POS_ENHASH) << ": " << hashsize << " bytes" << std::endl;
            tt += (long long)(hashsize * (unsigned long long)n);
        }
	}

	printf( "Index size=%0.3lfMB, Max label size=%lld, Avg label size=%0.3lf\n",
			(tt*1.0 + sizeof(BPLabel)*1.0*nown)/(1024*1024), s, tt*0.25/n);

	// for( int i = 0; i < n; ++i ) delete[] pos[i];
	// delete[] pos;
	
	// see_hash();
	init_query();
}

void DisOracle::construct_bp_label(int max_range) {
	printf( "Constructing BP Label...\n" );
	label_bp = new BPLabel[nown];
	usd_bp = new bool[n];
	memset( usd_bp, 0, sizeof(bool) * n );
	vector<int> v_vs[N_ROOTS];

	int r = 0;
	for (int i_bpspt = 0; i_bpspt < N_ROOTS; ++i_bpspt) {
		while (r < nown && usd_bp[r]) ++r;
		if (r == nown) {
			for (int v = 0; v < nown; ++v) label_bp[v].bpspt_d[i_bpspt] = MAXD;
			continue;
		}
		usd_bp[r] = true;
		v_vs[i_bpspt].push_back(r);
		int ns = 0;
		for (int i = 0; i < deg[r]; ++i) {
			int v = con[r][i];
			if (!usd_bp[v]) {
				usd_bp[v] = true;
				v_vs[i_bpspt].push_back(v);
				if (++ns == 64) break;
			}
		}
	}

	int n_threads = 1;
	#pragma omp parallel
	{
		if(omp_get_thread_num() == 0) n_threads = omp_get_num_threads();
	}
	if( n_threads > MAX_BP_THREADS ) omp_set_num_threads(MAX_BP_THREADS);

	#pragma omp parallel
	{
		int pid = omp_get_thread_num(), np = omp_get_num_threads();
		if( pid == 0 ) printf( "n_threads_bp = %d\n", np );
		vector<uint8_t> tmp_d(nown);
		vector<pair<uint64_t, uint64_t> > tmp_s(nown);
		vector<int> que(nown);
		//vector<pair<int, int> > sibling_es(m/2);
		vector<pair<int, int> > child_es(m/2);

		for (int i_bpspt = pid; i_bpspt < N_ROOTS; i_bpspt+=np) {
			printf( "[%d]", i_bpspt );

			if( v_vs[i_bpspt].size() == 0 ) continue;
			fill(tmp_d.begin(), tmp_d.end(), MAXD);
			fill(tmp_s.begin(), tmp_s.end(), make_pair(0, 0));

			r = v_vs[i_bpspt][0];
			int que_t0 = 0, que_t1 = 0, que_h = 0;
			que[que_h++] = r;
			tmp_d[r] = 0;
			que_t1 = que_h;

			for( size_t i = 1; i < v_vs[i_bpspt].size(); ++i) {
				int v = v_vs[i_bpspt][i];
				que[que_h++] = v;
				tmp_d[v] = 1;
				tmp_s[v].first = 1ULL << (i-1);
			}

			for (int d = 0; que_t0 < que_h; ++d) {
				if (d+1>max_range) break;
				//int num_sibling_es = 0;
				int num_child_es = 0;

				for (int que_i = que_t0; que_i < que_t1; ++que_i) {
					int v = que[que_i];

					for (int i = 0; i < deg[v]; ++i) {
						int tv = con[v][i];
						int td = d + 1;
						if (d == tmp_d[tv]) {
							if (v < tv) {
								//sibling_es[num_sibling_es].first  = v;
								//sibling_es[num_sibling_es].second = tv;
								//++num_sibling_es;
								tmp_s[v].second |= tmp_s[tv].first;
								tmp_s[tv].second |= tmp_s[v].first;
							}
						} else if( d < tmp_d[tv]) {
							if (tmp_d[tv] == MAXD) {
								que[que_h++] = tv;
								tmp_d[tv] = td;
							}
							child_es[num_child_es].first  = v;
							child_es[num_child_es].second = tv;
							++num_child_es;
							//tmp_s[tv].first  |= tmp_s[v].first;
							//tmp_s[tv].second |= tmp_s[v].second;
						}
					}
				}

				/*for (int i = 0; i < num_sibling_es; ++i) {
					int v = sibling_es[i].first, w = sibling_es[i].second;
					tmp_s[v].second |= tmp_s[w].first;
					tmp_s[w].second |= tmp_s[v].first;
				}*/

				for (int i = 0; i < num_child_es; ++i) {
					int v = child_es[i].first, c = child_es[i].second;
					tmp_s[c].first  |= tmp_s[v].first;
					tmp_s[c].second |= tmp_s[v].second;
				}

				que_t0 = que_t1;
				que_t1 = que_h;
			}

			for (int v = 0; v < nown; ++v) {
				label_bp[v].bpspt_d[i_bpspt] = tmp_d[v];
				label_bp[v].bpspt_s[i_bpspt][0] = tmp_s[v].first;
				label_bp[v].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
			}
		}
	}
	omp_set_num_threads(n_threads);
	printf( "\nBP Label Constructed, bp_size=%0.3lfMB\n", sizeof(BPLabel)*nown/(1024.0*1024.0));
}

void DisOracle::see_labels() {
	std::vector<int> cnts;
	for (int i=0; i<MAXDIS; i++) cnts.push_back(0);
	for (int i=0; i<n; i++) {
		for (int j=0; j<MAXDIS; j++) {
			cnts[j] += (pos[i][j+1]-pos[i][j]);
		}
	}
	for (int j=0; j<MAXDIS; j++) std::cout << "dis i:" << cnts[j]/n << std::endl;

}

int DisOracle::query_by_bp(int u, int v) {
	BPLabel &idx_u = label_bp[u], &idx_v = label_bp[v];
	int d = MAXD;
	for (int i = 0; i < N_ROOTS; ++i) {
		int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
		if (td - 2 <= d)
			td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
					((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) | (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0])) ? -1 : 0;
		if (td < d) d = td;
	}
	return d;
}

bool DisOracle::query_by_bp_es(int u, int v, int search_k) {
	BPLabel &idx_u = label_bp[u], &idx_v = label_bp[v];
	for (int i = 0; i < N_ROOTS; ++i) {
		int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
		if (td <= search_k) return true;
		// V1: directly compute
		if (td-2 <= search_k)
			td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
				((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) | (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0])) ? -1 : 0;
		// V2: judge, but should record minimum d
		// if (td - 2 <= d)
		if (td <= search_k) return true;
	}
	return false;
}

bool DisOracle::prune_by_bp(int u, int v, int d) {
	BPLabel &idx_u = label_bp[u], &idx_v = label_bp[v];
	for (int i = 0; i < N_ROOTS; ++i) {
		int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
		if (td - 2 <= d)
			td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
					((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) | (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0])) ? -1 : 0;
		if (td <= d) return true;
	}
	return false;
}

void DisOracle::save_idx(string path) {
	printf( "Saving index...\n" );
	FILE *fout = fopen( (path+"index-dis.bin").c_str(), "wb" );
	fwrite(&n, sizeof(int), 1, fout);
	for(int i = 0; i < n; ++i) {
		int len = (int) label[i].size();
		fwrite(&len, sizeof(int), 1, fout);
	}
	unsigned *s = new unsigned[n];
	for(int i = 0; i < n; ++i) {
		int len = (int) label[i].size();
		for(int j = 0; j < len; ++j) s[j] = label[i][j];
		fwrite(s, sizeof(unsigned), len, fout);
	}
	fwrite(&MAXMOV, sizeof(int), 1, fout);
	if(LOAD_INDEP_LABEL) memset( is_indep, 0, sizeof(bool) * n );
	fwrite(is_indep, sizeof(bool), n, fout);
	fwrite(usd_bp, sizeof(bool), n, fout);

	int n_roots = N_ROOTS;
	fwrite(&n_roots, sizeof(int), 1, fout);
	fwrite(&nown, sizeof(int), 1, fout);
	fwrite(label_bp, sizeof(BPLabel), nown, fout);
	fclose(fout);
	delete[] s;
}

void DisOracle::init_query() {
	t = 0;
	last_t = new tint[n];
	memset( last_t, 0, sizeof(tint) * n );
	dis = new char[n];
}

void DisOracle::load_graph(string path) {
	printf( "loading graph: %s\n", path.c_str() );
	long long p = 0;
	FILE* fin = fopen( (path+"graph-dis.bin").c_str(), "rb" );
	fread( &n, sizeof(int), 1, fin );
	fread( &m, sizeof(long long), 1, fin );
	deg = new int[n]; dat = new int[m]; con = new int*[n]; nid = new int[n];
	fread( deg, sizeof(int), n, fin );
	fread( dat, sizeof(int), m, fin );
	fread( nid, sizeof(int), n, fin );
	fclose(fin);
	for( int i = 0; i < n; ++i ) {con[i] = dat+p; p+= deg[i];}
	nown = n-1; while(nown>=0 && deg[nown] == 0) --nown;
	nown += 1; if( nown < 2 ) nown = 2;
	printf( "graph loaded, n=%d, m=%lld\n", n, m);
}

void DisOracle::load_idx(string path) {
	printf( "Loading index...\n" );
	FILE *fin = fopen( (path+"index-dis.bin").c_str(), "rb" );
	fread(&n, sizeof(int), 1, fin);
	int *len = new int[n];
	fread(len, sizeof(int), n, fin);
	long long index_size = 0;
	for( int i = 0; i < n; ++i ) index_size += len[i];

	label = new vector<unsigned>[n];
	unsigned *s = new unsigned[n];
	for( int i = 0; i < n; ++i ) {
		fread(s, sizeof(unsigned), len[i], fin);
		label[i].reserve(len[i]);
		label[i].assign(s, s + len[i]);
	}
	delete[] s;
	delete[] len;


	fread(&MAXMOV, sizeof(unsigned), 1, fin);
	MAXDIS = 1<<MAXMOV; MASK = MAXDIS-1;
	is_indep = new bool[n];
	fread(is_indep, sizeof(bool), n, fin);

	usd_bp = new bool[n];
	fread(usd_bp, sizeof(bool), n, fin);

	int n_roots;
	fread(&n_roots, sizeof(int), 1, fin);
	if( n_roots != N_ROOTS ) printf( "Error! n_roots=%d,N_RROTS=%d\n", n_roots, N_ROOTS);
	fread(&nown, sizeof(int), 1, fin);
	label_bp = new BPLabel[nown];
	fread(label_bp, sizeof(BPLabel), nown, fin);
	fclose(fin);

	index_size = index_size * 4 + sizeof(BPLabel) * nown;
	printf( "index_size=%0.3lf GB\n", index_size/(1024.0*1024.0*1024.0));

	load_graph(path);
	init_query();
}

int DisOracle::query_by_nid(int u, int v) {
	unsigned lu = (unsigned)label[u].size(), lv = (unsigned)label[v].size(), dis = MAXD;
	for( int i = 0, j = 0; i < lu && j < lv; ++i ) {
		for( ; j < lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; ++j ) ;
		if( j < lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV) dis = min(dis, (label[u][i]&MASK) + (label[v][j]&MASK));
	}
	return dis;
}

// bool DisOracle::query_by_nid_es_DEP(int u, int v, int search_k) {
// 	unsigned lu = (unsigned)label[u].size(), lv = (unsigned)label[v].size(), dis = MAXD;
// 	for( int i = 0, j = 0; i < lu && j < lv; ++i ) {
// 		for( ; j < lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; ++j ) ;
// 		if( j < lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV && ((label[u][i]&MASK) + (label[v][j]&MASK))<=search_k) return true;
// 	}
// 	return false;
// }

// bool DisOracle::query_by_nid_esV0(int u, int v, int search_k) {
// V0: only with early stop
// 	unsigned lu = (unsigned)pos[u][MAXDIS-1], lv = (unsigned)pos[v][MAXDIS-1];
	
// 	for( int i = 0, j = 0; i < lu && j < lv; ++i ) {
// 		for( ; j < lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; ++j ) ;
// 		if( j < lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV && ((label[u][i]&MASK) + (label[v][j]&MASK))<=search_k) return true;
// 	}
	
// 	for (int i=pos[v][MAXDIS-1]; i<pos[v][MAXDIS]; i++) {
// 		if (label[v][i]>>MAXMOV == u) return true;
// 	}
// 	for (int i=pos[u][MAXDIS-1]; i<pos[u][MAXDIS]; i++) {
// 		if (label[u][i]>>MAXMOV == v) return true;
// 	}
// 	return false;
// }

// bool DisOracle::query_by_nid_esV1(int u, int v, int search_k) {
// 	// V1: early stop + range-aware filter
// 	for (int ks = 1; ks<=search_k; ks++) {
// 		// std::cout << "ks" << ks << std::endl;
// 		unsigned lu = (unsigned)pos[u][ks], lv = (unsigned)pos[v][search_k - ks];
// 		for (int i=(unsigned) pos[u][ks-1], j=0; i<lu && j<lv; i++) {
// 			// std::cout << i << "," << j << std::endl;
// 			for (; j<lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; j++) ;
// 			if (j<lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV) return true;
// 		}
// 	}
// 	return false;
// }

void DisOracle::init_query_node(int q) {

}


bool DisOracle::query_by_nid_es(int u, int v, int search_k) {
	++t;
		if (t == MAXT) {
			memset(last_t, 0, sizeof(tint)*n);
			t=1;
		}
	
	int ktmp = min(search_k, POS_ENHASH-1);

	for (int ks = 0; ks <=ktmp; ks++) {
		
		int ru = pos[u][ks]; //(ks==search_k)?(int) label[u].size():pos[u][ks];
		for (int i=0; i<ru; i++) {
			int w = label[u][i] >> MAXMOV;
			char d = label[u][i]&MASK;
			last_t[w] = t; dis[w] = d;
		}
		int rks = std::max(0, std::min(ktmp, search_k-ks));
		// std::cout << "ks:" << ks << " rks:" << rks << std::endl;
		int lv = (rks==0)?0:pos[v][rks-1];
		int rv = pos[v][rks]; //(rks==search_k)?(int) label[v].size():pos[v][rks];
		for (int i=lv; i < rv; i++) {
			int w = label[v][i]>>MAXMOV;
			char d = label[v][i]&MASK;
			if (last_t[w] == t && (char)(d+dis[w])<=search_k) return true;
		}
	}
	// return false;

	// int ktmp = min(search_k, POS_ENHASH);
	// for (int ks = 1; ks<ktmp; ks++) {
	// 	// std::cout << "ks" << ks << std::endl;
	// 	int ls = min(search_k-ks, POS_ENHASH-1);
	// 	unsigned lu = (unsigned)pos[u][ks], lv = (unsigned)pos[v][ls];
	// 	for (int i=(unsigned) pos[u][ks-1], j=0; i<lu && j<lv; i++) {
	// 		// std::cout << i << "," << j << std::endl;
	// 		for (; j<lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; j++) ;
	// 		if (j<lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV) return true;
	// 	}
	// }

	if (POS_ENHASH<=search_k) {
		for (int i=POS_ENHASH; i<=search_k; i++) {
			int rr = search_k-i;
			if (rr>=0 && hashes[u][i-POS_ENHASH].isIntersect(label[v], 0, pos[v][0])) {
				return true;
			}
			for (int j=1; j<=rr; j++) {
				if (j<POS_ENHASH &&
				hashes[u][i-POS_ENHASH].isIntersect(label[v], pos[v][j-1], pos[v][j]-pos[v][j-1])) return true;
				else if (j>=POS_ENHASH && hashes[u][i-POS_ENHASH].isIntersect(hashes[v][j-POS_ENHASH])) return true;
			}
		}  

		
		for (int j=POS_ENHASH; j<=search_k; j++) {
			int rr = min(search_k-j, POS_ENHASH-1);
			if (rr>=0 && hashes[v][j-POS_ENHASH].isIntersect(label[u], 0, pos[u][0])) return true;
			for (int i=1; i<=rr; i++) {
				if (hashes[v][j-POS_ENHASH].isIntersect(label[u], pos[u][i-1], pos[u][i]-pos[u][i-1])) return true;
			}
		}
	}
	return false;
}



int DisOracle::query_by_t(int u, int v) {
	++t;
	if(t == MAXT) {
		memset(last_t, 0, sizeof(tint) * n);
		t=1;
	}
	char mind = MAXD;
	if( !is_indep[u] ) {
		int lu = (int) label[u].size();
		for( int i = 0; i < lu; ++i ) {
			int w = label[u][i]>>MAXMOV;
			char d = label[u][i]&MASK;
			last_t[w] = t; dis[w] = d;
		}
	} else {
		for( int i = 0; i < deg[u]; ++i ) {
			int x = con[u][i];
			int lx = (int) label[x].size();
			for( int j = 0; j < lx; ++j ) {
				int w = label[x][j]>>MAXMOV;
				char d = label[x][j]&MASK;
				if( last_t[w] != t ) {last_t[w] = t; dis[w] = d+1;} else {dis[w]=min(dis[w],(char)(d+1));}
			}
		}
	}
	if( !is_indep[v] ) {
		int lv = (int) label[v].size();
		for( int i = 0; i < lv; ++i ) {
			int w = label[v][i]>>MAXMOV;
			char d = label[v][i]&MASK;
			if(last_t[w] == t) mind = min(mind, (char)(d+dis[w]));
		}
	} else {
		for( int i = 0; i < deg[v]; ++i ) {
			int x = con[v][i];
			int lx = (int) label[x].size();
			for( int j = 0; j < lx; ++j ) {
				int w = label[x][j]>>MAXMOV;
				char d = label[x][j]&MASK;
				if( last_t[w] == t ) mind = min(mind, (char)(d+1+dis[w]));
			}
		}
	}
	return mind;
}


bool DisOracle::query_by_t_es(int u, int v, int search_k) {
	++t;
	if(t == MAXT) {
		memset(last_t, 0, sizeof(tint) * n);
		t=1;
	}
	char mind = MAXD;
	if( !is_indep[u] ) {
		int lu = (int) label[u].size();
		for( int i = 0; i < lu; ++i ) {
			int w = label[u][i]>>MAXMOV;
			char d = label[u][i]&MASK;
			last_t[w] = t; dis[w] = d;
		}
	} else {
		for( int i = 0; i < deg[u]; ++i ) {
			int x = con[u][i];
			int lx = (int) label[x].size();
			for( int j = 0; j < lx; ++j ) {
				int w = label[x][j]>>MAXMOV;
				char d = label[x][j]&MASK;
				if( last_t[w] != t ) {last_t[w] = t; dis[w] = d+1;} else {dis[w]=min(dis[w],(char)(d+1));}
			}
		}
	}
	if( !is_indep[v] ) {
		int lv = (int) label[v].size();
		for( int i = 0; i < lv; ++i ) {
			int w = label[v][i]>>MAXMOV;
			char d = label[v][i]&MASK;
			if(last_t[w] == t && (char)(d+dis[w])<=search_k) return true;
		}
	} else {
		for( int i = 0; i < deg[v]; ++i ) {
			int x = con[v][i];
			int lx = (int) label[x].size();
			for( int j = 0; j < lx; ++j ) {
				int w = label[x][j]>>MAXMOV;
				char d = label[x][j]&MASK;
				if( last_t[w] == t && (char)(d+1+dis[w])<=search_k) return true;
			}
		}
	}
	return false;
}

int DisOracle::query(int u, int v) {
	if( u == v ) return 0;
	int type = 0;
	u = nid[u]; v = nid[v];
	if(u < 0) {u = -u-1; type = 1;} else if(u >= MAXN) {u = u-MAXN; type = 2;}
	if(v < 0) {v = -v-1; type = 1;} else if(v >= MAXN) {v = v-MAXN; type = 2;}
	if(u == v) return type == 1 ? (deg[u] == 0 ? MAXD : 2) : 1;
	if( u >= nown || v >= nown ) return MAXD;
	int d = query_by_bp(u,v);
	if(!is_indep[u] && !is_indep[v]) return min(d,query_by_nid(u, v));
	if( is_indep[u] && !is_indep[v] ) swap(u,v);
	if( is_indep[u] && is_indep[v] && label[u].size() > label[v].size() ) swap(u,v);
	return min(d, query_by_t(u, v));
}

bool DisOracle::query(int u, int v, int search_k) {
	if( u == v ) return 0;
	// if (u == v) return true;
	int type = 0;
	u = nid[u]; v = nid[v];
	if(u < 0) {u = -u-1; type = 1;} else if(u >= MAXN) {u = u-MAXN; type = 2;}
	if(v < 0) {v = -v-1; type = 1;} else if(v >= MAXN) {v = v-MAXN; type = 2;}
	if(u == v) return type == 1 ? (deg[u] == 0 ? MAXD : 2) : 1;
	if( u >= nown || v >= nown ) return MAXD;
	if (query_by_bp_es(u,v,search_k)) return true;
	if(!is_indep[u] && !is_indep[v]) return query_by_nid_es(u, v, search_k);
	if( is_indep[u] && !is_indep[v] ) swap(u,v);
	if( is_indep[u] && is_indep[v] && label[u].size() > label[v].size() ) swap(u,v);
	return query_by_t_es(u, v, search_k);
}

DisOracle::~DisOracle() {
	if(label) delete[] label; if(nid) delete[] nid; if(is_indep) delete[] is_indep; if(last_t) delete[] last_t; if(dis) delete[] dis;
	if(con) delete[] con; if(dat) delete[] dat; if(deg) delete[] deg;
	if( label_bp ) delete[] label_bp; if( usd_bp ) delete[] usd_bp;
}

// }
#endif /* DIS_H_ */
