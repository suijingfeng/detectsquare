#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define PI 3.1415926535
using namespace std;
struct node
{
    int x,y;
};

//存入的所有的点
node vex[1000];

int xx,yy;

bool cmp1(node a, node b)//排序找第一个点
{
    if(a.y == b.y)
        return a.x < b.x;
    else
        return a.y < b.y;
}

// (b - a) x ( c - a)
int inline cross(node a, node b, node c)
{
    return (b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y);
}

// distance
double inline dis(node a, node b)
{
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

bool cmp2(node a, node b)//极角排序另一种方法，速度快
{
    if(atan2(a.y-yy, a.x-xx) != atan2(b.y-yy,b.x-xx))
        return (atan2(a.y-yy,a.x-xx))<(atan2(b.y-yy,b.x-xx));
    return a.x<b.x;
}


bool cmp(node a, node b)//极角排序
{
    int m=cross(vex[0],a,b);
    if(m>0)
        return 1;
    else if(m==0&&dis(vex[0],a)-dis(vex[0],b)<=0)
        return 1;
    else return 0;
    /*if(m==0)
        return dis(vex[0],a)-dis(vex[0],b)<=0?true:false;
    else
        return m>0?true:false;*/
}

int main()
{
    int t;
    scanf("%d", &t);
    if(t < 3)
    {   
        printf("Inputed points are not enough!\n");
        return 0;
    }
    
    //凸包中所有的点
    node stackk[1000];
    memset(stackk, 0, sizeof(stackk));

    int i;
    for(i=0; i<t; i++)
    {
        scanf("%d%d", &vex[i].x, &vex[i].y);
    }

    // graham
    {
        sort(vex, vex+t, cmp1);

        //将凸包中的前两个点存入凸包的结构体中
        stackk[0] = vex[0];
        stackk[1] = vex[1];

        xx = stackk[0].x;
        yy = stackk[0].y;
        sort(vex+1,vex+t,cmp2);//cmp2是更快的，cmp更容易理解
        int top=1;//最后凸包中拥有点的个数
        for(i=2; i<t; i++)
        {
            while( cross(stackk[top-1],stackk[top],vex[i]) < 0 )  //控制<0或<=0可以控制重点，共线的，具体视题目而定。
                top--;
            stackk[++top]=vex[i];
        }
        for(i=1; i<=top; i++)//输出凸包上的点
            cout<<stackk[i].x<<" "<<stackk[i].y<<endl;
    }

}
