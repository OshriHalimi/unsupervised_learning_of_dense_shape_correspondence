/* Compute the Gaussian Curvature of a triangulated mesh
 Input:	x,y,z = Array of coordinates
 tri = triangulation of the mesh
 Output	g = Gaussian Curvature
 2011 Yonathan Aflalo
 */




#include <iostream>
#include <list>
#include <vector>
#include "mex.h"
#include <math.h>
#include "matrix.h"
#include <string.h>
#include <set>

using namespace std;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define PI 3.14159265
extern int main();

int sub2ind(int m,int i,int j);
double calcangle(double *x,double *y,double *z,int a,int b, int c);
double calcarea(double *x,double *y,double *z,int a,int b, int c);
void Compute_Gaussian_Curvature(double *g,double *x,double *y,double *z,double *tri,const int m,const size_t size_tri, set<int> *neighboor,set<int> *triangles);
void ind2sub(int m,int ind,int *value);
double len(int a,int b,double *x,double *y,double *z);

class edge{
public:
	int e1;
	int e2;
	set<int> tri;
	int pos;
	bool operator ==(const edge& b) const{
		return (e1 == b.e1 && e2 == b.e2) || (e1 == b.e2 && e2 == b.e1);
	}
    
	edge(int e1, int e2, int pos) : e1(e1),e2(e2),pos(pos){
	}
    
	edge(int e1, int e2) : e1(e1),e2(e2){
	}
    
	bool operator < (const edge& b) const{
		return  pow((double)e1,4)+pow((double)e2,4)<pow((double)b.e1,4)+pow((double)b.e2,4) ; 
	}
	int intersect(const edge& b){
		if(e1 == b.e1)
			return e1;
		if(e1 == b.e2)
			return e1;
		if(e2 == b.e1)
			return e2;
		if(e2 == b.e2)
			return e2;
		return -1;
	}
};
const int numInputArgs  = 5;
const int numOutputArgs1 = 1;
const int numOutputArgs2 = 2;
const int numOutputArgs3 = 3;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != numInputArgs)
		mexErrMsgTxt("Incorrect number of input arguments");
	if (nlhs != numOutputArgs1 && nlhs != numOutputArgs2 && nlhs != numOutputArgs3)
		mexErrMsgTxt("Incorrect number of output arguments");
	double* x = mxGetPr(prhs[0]);
	double* y = mxGetPr(prhs[1]);
	double* z = mxGetPr(prhs[2]);
	double* tri_temp = mxGetPr(prhs[3]);
    size_t size_tri=mxGetM(prhs[3]);
	mxArray *plhstri[1];
    mxArray *tri_temp_pr[1];
    tri_temp_pr[0] = mxCreateDoubleMatrix(size_tri, 3, mxREAL);
    double* tri_temp2 = mxGetPr(tri_temp_pr[0]);
    for(int i=0;i<3*size_tri;i++)
        tri_temp2[i] = tri_temp[i];
	mexCallMATLAB(1, plhstri, 1, &tri_temp_pr[0], "transpose");
	double* tri = mxGetPr(plhstri[0]);
	size_t m=mxGetM(prhs[0]);
	size_t n=mxGetN(prhs[0]);
	for (int i = 1;i < 3;i++)
		if (mxGetM(prhs[i]) != m || mxGetN(prhs[i]) != n)
			mexErrMsgTxt("Incorrect dimension");
	if(m == 1 && n != 1){
		m=n;
		n=1;
    }
    mxArray *curv = mxCreateDoubleMatrix(m, 1, mxREAL);
    double *g=mxGetPr(curv);
    set<int> *neighboor=new set<int>[m];
	set<int> *triangles=new set<int>[m];
	for(int i=0;i<(int)size_tri;i++){
		int a=(int) floor(tri[sub2ind(3,i,0)])-1;
		int b=(int) floor(tri[sub2ind(3,i,1)])-1;
		int c=(int) floor(tri[sub2ind(3,i,2)])-1;
		neighboor[a].insert(b);
		neighboor[a].insert(c);
		triangles[a].insert(i);
		neighboor[b].insert(a);
		neighboor[b].insert(c);
		triangles[b].insert(i);
		neighboor[c].insert(b);
		neighboor[c].insert(a);
		triangles[c].insert(i);
	}
    Compute_Gaussian_Curvature(g,x,y,z,tri,m,size_tri,neighboor,triangles);
    mwSize dims[2] = {1, size_tri};
    plhs[0]=mxCreateCellArray(2, dims);
    double* cond_scale = mxGetPr(prhs[4]);
    double* g_tri_out;
    if(nlhs == numOutputArgs2 || nlhs == numOutputArgs3){
        plhs[1]=mxCreateDoubleMatrix(size_tri,1,mxREAL);
        g_tri_out=mxGetPr(plhs[1]);
    }
    if(nlhs == numOutputArgs3){
        plhs[2]=curv;
    }

    bool cond = (bool) cond_scale[0];
    for (int i = 0; i< size_tri; i++) {
        int a=(int) floor(tri[sub2ind(3,i,0)])-1;
		int b=(int) floor(tri[sub2ind(3,i,1)])-1;
		int c=(int) floor(tri[sub2ind(3,i,2)])-1;
        set<int> local_neigh;
        for (set<int>::iterator it=neighboor[a].begin(); it != neighboor[a].end(); it++) {
            local_neigh.insert(*it);
        }
        for (set<int>::iterator it=neighboor[b].begin(); it != neighboor[b].end(); it++) {
            local_neigh.insert(*it);
        }
        for (set<int>::iterator it=neighboor[c].begin(); it != neighboor[c].end(); it++) {
            local_neigh.insert(*it);
        }
        local_neigh.erase(a);
        local_neigh.erase(b);
        local_neigh.erase(c);
        double g_tri_neigh=0;
        
        for (set<int>::iterator it=local_neigh.begin(); it != local_neigh.end(); it++) {
            g_tri_neigh += g[*it];
        }
        g_tri_neigh/=(double) local_neigh.size();
        double g_tri=1.0/3.0*fabs(g[a]+g[b]+g[c]);
        g_tri = 0.75*g_tri+0.25*fabs(g_tri_neigh);
        if(nlhs == numOutputArgs2)
            g_tri_out[i]=g_tri;
        if(fabs(g_tri)<1e-20){
            g_tri=1e-20;
        }
        if (!cond) {
            g_tri = 1;
        }
        mxArray *G_temp;
        G_temp = mxCreateDoubleMatrix(2,2,mxREAL);
        double *mat_temp = mxGetPr(G_temp);
        double la = len(b,c,x,y,z);
        double lb = len(c,a,x,y,z);
        double lc = len(b,a,x,y,z);
        double g1 = g_tri*lc*lc;
        double g2 = g_tri*(lb*lb+lc*lc-la*la)/2.0;
        double g3 = g_tri*lb*lb;
        mat_temp[0]=g1;
        mat_temp[1]=g2;
        mat_temp[2]=g2;
        mat_temp[3]=g3;
        mxSetCell(plhs[0],i,G_temp);
    }
    mxDestroyArray(plhstri[0]);
    mxDestroyArray(tri_temp_pr[0]);
    if (nlhs != numOutputArgs3) {
        mxDestroyArray(curv);
    }
//        delete[] triangles;
    delete[] neighboor;
    
}

double len(int a,int b,double *x,double *y,double *z){
    return sqrt((x[a]-x[b])*(x[a]-x[b]) + (y[a]-y[b])*(y[a]-y[b]) + (z[a]-z[b])*(z[a]-z[b]));
    
}

void Compute_Gaussian_Curvature(double *g,double *x,double *y,double *z,double *tri,const int m,const size_t size_tri, set<int> *neighboor,set<int> *triangles){
		int k=0;
	for(int j=0;j<(int) m;j++){
        double factor = 2*PI;
        double angle = 0;
        double area = 0;
        for (set<int>::iterator curr_tri=triangles[j].begin(); curr_tri != triangles[j].end(); curr_tri++){
            int index_tri = *curr_tri;
            int a=(int) floor(tri[sub2ind(3,index_tri,0)])-1;
            int b=(int) floor(tri[sub2ind(3,index_tri,1)])-1;
            int c=(int) floor(tri[sub2ind(3,index_tri,2)])-1;
            if (j == b){
                b = a;
                a = j;
            }
            if (j == c){
                c = a;
                a = j;
            }
            angle += calcangle(x, y, z, a, b, c);
            area += calcarea(x, y, z, a, b, c);
        }
        
		for (set<int>::iterator it=neighboor[j].begin() ; it != neighboor[j].end(); it++ ){
            int num_edge = 0;
			int current_point=*it;
			set<int> lst_triangle1=triangles[j];
			set<int> lst_triangle2=triangles[current_point];
			for (set<int>::iterator it2 = lst_triangle1.begin() ; it2 != lst_triangle1.end(); it2++ ){
				set<int>::iterator temp = lst_triangle2.find(*it2);
				if(temp != lst_triangle2.end()){
                    num_edge++;
				}
			}
			if(num_edge == 1){
                factor = PI;
			}
		}
        g[j] = fabs(factor - angle)/area;
	}
}


void ind2sub(int m,int ind,int *value){
	value[0]=(ind)/m;
	value[1]=ind%m;  
}


int sub2ind(int m,int i,int j){
	return i*m+j;
}

double calcangle(double *x,double *y,double *z,int a,int b, int c){
	double cosalpha=((x[b]-x[a])*(x[c]-x[a]) + (y[b]-y[a])*(y[c]-y[a]) + (z[b]-z[a])*(z[c]-z[a])) /
    sqrt( ((x[b]-x[a])*(x[b]-x[a]) + (y[b]-y[a])*(y[b]-y[a]) + (z[b]-z[a])*(z[b]-z[a]) ) * ((x[c]-x[a])*(x[c]-x[a]) + (y[c]-y[a])*(y[c]-y[a]) + (z[c]-z[a])*(z[c]-z[a])));
	return acos(cosalpha);
}

double calcarea(double *x,double *y,double *z,int a,int b, int c){
	double ttx=(y[b]-y[c])*(z[a]-z[c])-(z[b]-z[c])*(y[a]-y[c]);
    double tty=(z[b]-z[c])*(x[a]-x[c])-(x[b]-x[c])*(z[a]-z[c]);
    double ttz=(x[b]-x[c])*(y[a]-y[c])-(y[b]-y[c])*(x[a]-x[c]);
    return sqrt(ttx*ttx+tty*tty+ttz*ttz);
}
