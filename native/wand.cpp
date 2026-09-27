// Positive-weight exact WAND retrieval over sorted inverted postings.
// No BM25 shortlist, vocabulary hashing, dropped query terms, or dense Q x N matrix.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>
#include <omp.h>
struct It { const int32_t* d; const float* v; int64_t p,e; double q,ub; };
extern "C" int er_wand(
 const int64_t* pp, const int32_t* docs, const float* vals, const float* maxima,
 const int64_t* qp, const int32_t* qt, const float* qv,
 int64_t nq, int k, int threads, int64_t* outid, float* outscore) {
 if(k<=0 || nq<0 || threads<=0) return 1;
 #pragma omp parallel for schedule(dynamic,1) num_threads(threads)
 for(int64_t r=0;r<nq;r++) {
  for(int j=0;j<k;j++){outid[r*k+j]=-1;outscore[r*k+j]=0;}
  std::vector<It> a; a.reserve(qp[r+1]-qp[r]);
  for(int64_t t=qp[r];t<qp[r+1];t++) {
   int f=qt[t]; if(qv[t]>0 && pp[f]<pp[f+1])
    a.push_back({docs,vals,pp[f],pp[f+1],double(qv[t]),double(qv[t])*double(maxima[f])});
  }
  using Hit=std::pair<double,int64_t>; // (score, -docid); minimum is worst retained result
  std::priority_queue<Hit,std::vector<Hit>,std::greater<Hit>> heap;
  while(!a.empty()) {
   std::sort(a.begin(),a.end(),[](const It&x,const It&y){return x.d[x.p]<y.d[y.p];});
   const double threshold=heap.size()==size_t(k)?heap.top().first:0.;
   double bound=0; size_t pivot=0;
   for(;pivot<a.size();pivot++){bound+=a[pivot].ub;if(bound+1e-10>=threshold)break;}
   if(pivot==a.size())break;
   const int32_t pd=a[pivot].d[a[pivot].p];
   if(a[0].d[a[0].p]==pd) {
    double s=0;
    for(auto &it:a){if(it.d[it.p]==pd){s+=it.q*it.v[it.p];it.p++;}}
    if(s>0){Hit h={s,-int64_t(pd)};if(heap.size()<size_t(k))heap.push(h);else if(h>heap.top()){heap.pop();heap.push(h);}}
   } else {
    for(size_t i=0;i<pivot;i++) {
     auto &it=a[i]; if(it.d[it.p]<pd)
      it.p=std::lower_bound(it.d+it.p,it.d+it.e,pd)-it.d;
    }
   }
   a.erase(std::remove_if(a.begin(),a.end(),[](const It&it){return it.p==it.e;}),a.end());
  }
  std::vector<Hit> h;
  while(!heap.empty()){h.push_back(heap.top());heap.pop();}
  std::sort(h.begin(),h.end(),std::greater<Hit>());
  for(size_t j=0;j<h.size();j++){outid[r*k+j]=-h[j].second;outscore[r*k+j]=float(h[j].first);}
 }
 return 0;
}
