#include "grid_buffer.hpp"
#include <algorithm>
#include <limits>
#include <queue>
#include <tuple>
#include <numeric>
#include <utility>

namespace {
struct Grid {
 int w,h; double r; const bool* m;
 std::vector<double> d; std::vector<int> touched;
 Grid(Buffer& b,double resolution):w(b.view.shape[1]),h(b.view.shape[0]),r(resolution),m(b.data<bool>()),d(w*h,std::numeric_limits<double>::infinity()){}
 bool valid(int x,int y)const{return x>=0&&y>=0&&x<w&&y<h&&m[y*w+x];}
 void run(const std::vector<std::pair<int,double>>& starts,double bound){
  for (int p : touched) {
    d[p] = std::numeric_limits<double>::infinity();
  }
  touched.clear();
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;
  for(auto [p,c]:starts)if(p>=0&&p<w*h&&m[p]&&c<=bound&&c<d[p]){if(!std::isfinite(d[p]))touched.push_back(p);d[p]=c;q.emplace(c,p);}
  while(!q.empty()){
   auto [cost,p]=q.top();q.pop();if(cost!=d[p])continue;
   int x=p%w,y=p/w;
   for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
    if((!dx&&!dy)||!valid(x+dx,y+dy))continue;
    // Wheel local planner checks both orthogonal cells for diagonal moves.
    if(dx&&dy&&(!valid(x+dx,y)||!valid(x,y+dy)))continue;
    int n=(y+dy)*w+x+dx;double v=cost+r*(dx&&dy?std::sqrt(2.):1.);
    if(v<=bound+1e-10&&v<d[n]){if(!std::isfinite(d[n]))touched.push_back(n);d[n]=v;q.emplace(v,n);}
   }
  }
 }
 int index(const std::int64_t* p)const{return valid(p[0],p[1])?p[1]*w+p[0]:-1;}
};
PyObject* distances(PyObject*,PyObject* args){
 PyObject *m,*s,*c,*t,*o;double r,bound;
 if(!PyArg_ParseTuple(args,"OdOOdOO",&m,&r,&s,&c,&bound,&t,&o))return nullptr;
 try{Buffer mb(m,"?",2),sb(s,"l",2),cb(c,"d",1),tb(t,"l",2),ob(o,"d",1,true);
 if(sb.view.shape[1]!=2||tb.view.shape[1]!=2||cb.view.shape[0]!=sb.view.shape[0]||ob.view.shape[0]!=tb.view.shape[0])throw std::invalid_argument("metric dimensions");
 {ReleasedGIL release;Grid g(mb,r);std::vector<std::pair<int,double>> seeds;
 for(int i=0;i<sb.view.shape[0];++i)seeds.emplace_back(g.index(sb.data<std::int64_t>()+2*i),cb.data<double>()[i]);
 g.run(seeds,bound);for(int i=0;i<tb.view.shape[0];++i){int p=g.index(tb.data<std::int64_t>()+2*i);ob.data<double>()[i]=p<0?std::numeric_limits<double>::infinity():g.d[p];}}
 Py_RETURN_NONE;}catch(const std::exception&e){return error(e);}
}
PyObject* cover(PyObject*,PyObject* args){
 PyObject *m,*s,*o;double r,bound;if(!PyArg_ParseTuple(args,"OdOdO",&m,&r,&s,&bound,&o))return nullptr;
 try{Buffer mb(m,"?",2),sb(s,"l",2),ob(o,"l",2);std::vector<int> nodes;
 if(sb.view.shape[1]!=2||ob.view.shape[1]!=2)throw std::invalid_argument("cover cells require x/y pairs");
 {ReleasedGIL release;Grid g(mb,r);std::vector<bool> covered(g.w*g.h,false),selected(g.w*g.h,false);
 auto add=[&](int p){if(p<0||selected[p])return;selected[p]=true;nodes.push_back(p);g.run({{p,0.}},bound);for(int n:g.touched)covered[n]=true;};
 for(int i=0;i<sb.view.shape[0];++i)add(g.index(sb.data<std::int64_t>()+2*i));
 for(int i=0;i<ob.view.shape[0];++i){int p=g.index(ob.data<std::int64_t>()+2*i);if(p>=0&&!covered[p])add(p);}}
 PyObject* list=PyList_New(nodes.size());int w=mb.view.shape[1];for(size_t i=0;i<nodes.size();++i)PyList_SET_ITEM(list,i,Py_BuildValue("(ii)",nodes[i]%w,nodes[i]/w));return list;
 }catch(const std::exception&e){return error(e);}
}
PyObject* connections(PyObject*,PyObject* args){
 PyObject *m,*s;double r,bound;if(!PyArg_ParseTuple(args,"OdOd",&m,&r,&s,&bound))return nullptr;
 try{Buffer mb(m,"?",2),sb(s,"l",2);std::vector<std::tuple<int,int,double>> edges;
 {ReleasedGIL release;Grid g(mb,r);int count=sb.view.shape[0];std::vector<int> owner(g.w*g.h,-1);
 for(int i=0;i<count;++i){int p=g.index(sb.data<std::int64_t>()+2*i);if(p<0)throw std::invalid_argument("node outside traversable mask");owner[p]=i;}
 for(int i=0;i<count;++i){int p=g.index(sb.data<std::int64_t>()+2*i);g.run({{p,0.}},bound);for(int n:g.touched)if(owner[n]>i)edges.emplace_back(i,owner[n],g.d[n]);}}
 PyObject* list=PyList_New(edges.size());for(size_t i=0;i<edges.size();++i){auto[a,b,c]=edges[i];PyList_SET_ITEM(list,i,Py_BuildValue("(iid)",a,b,c));}return list;
 }catch(const std::exception&e){return error(e);}
}
PyObject* spanner(PyObject*,PyObject* args){
 int count;PyObject *e,*l;double stretch;
 if(!PyArg_ParseTuple(args,"iOOd",&count,&e,&l,&stretch))return nullptr;
 try{Buffer eb(e,"l",2),lb(l,"d",1);std::vector<int> kept;
 {ReleasedGIL release;
 std::vector<int> order(lb.view.shape[0]);std::iota(order.begin(),order.end(),0);
 auto edges=eb.data<std::int64_t>();auto lengths=lb.data<double>();
 std::sort(order.begin(),order.end(),[&](int a,int b){return std::tuple(lengths[a],edges[2*a],edges[2*a+1])<std::tuple(lengths[b],edges[2*b],edges[2*b+1]);});
 std::vector<std::vector<std::pair<int,double>>> adj(count);
 std::vector<double> d(count,std::numeric_limits<double>::infinity());std::vector<int> touched;
 for(int k:order){int a=edges[2*k],b=edges[2*k+1];double limit=stretch*lengths[k];
 for (int n : touched) {
   d[n] = std::numeric_limits<double>::infinity();
 }
 touched.clear();
 using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;
 d[a]=0;touched.push_back(a);q.emplace(0,a);bool found=false;
 while(!q.empty()){auto[c,n]=q.top();q.pop();if(c!=d[n])continue;if(n==b){found=true;break;}
 for(auto[v,len]:adj[n]){double next=c+len;if(next<=limit+1e-10&&next<d[v]){if(!std::isfinite(d[v]))touched.push_back(v);d[v]=next;q.emplace(next,v);}}}
 if(!found){kept.push_back(k);adj[a].emplace_back(b,lengths[k]);adj[b].emplace_back(a,lengths[k]);}}
 }
 PyObject* list=PyList_New(kept.size());for(size_t i=0;i<kept.size();++i)PyList_SET_ITEM(list,i,PyLong_FromLong(kept[i]));return list;
 }catch(const std::exception&e){return error(e);}
}
PyMethodDef methods[]={{"spanner",spanner,METH_VARARGS,nullptr},{"distances",distances,METH_VARARGS,nullptr},{"cover",cover,METH_VARARGS,nullptr},{"connections",connections,METH_VARARGS,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"lunar_drl_graph_native",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit_lunar_drl_graph_native(){return PyModule_Create(&module);}
