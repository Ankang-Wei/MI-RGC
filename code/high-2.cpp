// ï¿½ï¿½8ï¿½ï¿½ï¿½ß¼ï¿½ï¿½ï¿½Ñ¡ï¿½ï¿½ï¿½  È·ï¿½ï¿½ï¿½ï¿½Öµ
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <time.h>
//#include "omp.h"
using namespace std;
const int MAXN = 1e5 + 10;
const double fexp = 1e-6;
const int MERJIE = 3000;

bool matrixLAPP[MERJIE][MAXN];
int length;

struct node
{
    int x, y;
    double m;
};
node erjie[1440000];


bool cmp(node a, node b)
{
    if (fabs(a.m - b.m) < fexp && a.x == b.x) return a.y < b.y;
    if (fabs(a.m - b.m) < fexp) return a.x < b.x;
    return a.m > b.m;
}

bool judge(int X, int Y, int chosen, int j) ///ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
{
    switch(chosen)
    {
        case 1: ///x^y
            return matrixLAPP[X][j] && matrixLAPP[Y][j];
            break;
        case 2: /// -(x^y)
            return !(matrixLAPP[X][j] && matrixLAPP[Y][j]);
            break;
        case 3:/// x u y
            return matrixLAPP[X][j] || matrixLAPP[Y][j];
            break;
        case 4:/// -(x u y)
            return !(matrixLAPP[X][j] || matrixLAPP[Y][j]);
            break;
        case 51:/// x ^ !y
            return matrixLAPP[X][j] && (!matrixLAPP[Y][j]);
            /*if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 0) return 0;
            else if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 1) return 0;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 0) return 1;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 1) return 0;*/
            break;
        case 52:/// !x ^ y
            return (!matrixLAPP[X][j]) && matrixLAPP[Y][j];
            /*if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 0) return 0;
            else if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 1) return 1;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 0) return 0;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 1) return 0;*/
            break;
        case 62:/// x u !y
            return matrixLAPP[X][j] || (!matrixLAPP[Y][j]);
            /*if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 0) return 1;
            else if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 1) return 0;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 0) return 1;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 1) return 1;*/
            break;
        case 61:/// !x u y
            return (!matrixLAPP[X][j]) || matrixLAPP[Y][j];
            /*if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 0) return 1;
            else if (matrixLAPP[X][j] == 0 && matrixLAPP[Y][j] == 1) return 1;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 0) return 0;
            else if (matrixLAPP[X][j] == 1 && matrixLAPP[Y][j] == 1) return 1;*/
            break;
        case 7:/// !(x <> y)
            return !(matrixLAPP[X][j] == matrixLAPP[Y][j]);
            /*if (matrixLAPP[X][j] == matrixLAPP[Y][j]) return 0;
            return 1;*/
            break;
        case 8:/// x <> y
            return (matrixLAPP[X][j] == matrixLAPP[Y][j]);
            /*if (matrixLAPP[X][j] == matrixLAPP[Y][j]) return 1;
            return 0;*/
            break;
    }
}

double H(int X) ///Ò»ï¿½ï¿½ï¿½ï¿½Ï¢ï¿½ï¿½
{
    int cnt[2];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (matrixLAPP[X][j] == 0) {
            cnt[0]++;
        }
        else if (matrixLAPP[X][j] == 1) {
            cnt[1]++;
        }
    }
    double p[2];
    for (int i = 0; i < 2; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = double(cnt[i]) / length;
    }

    double ans = 0.0;
    for (int i = 0; i < 2; i++) {
        if (fabs(p[i] - 0.0) < 1e-6) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int X, int Y, int chosen) ///H(f(x, y))ï¿½ï¿½Ï¢ï¿½ï¿½
{
    int cnt[2];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (judge(X, Y, chosen, j) == 0) {
            cnt[0]++;
        }
        else if (judge(X, Y, chosen, j) == 1) {
            cnt[1]++;
        }
    }
    double p[2];
    memset(p, 0, sizeof(p));
    for (int i = 0; i < 2; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = double(cnt[i]) / length;
    }

    double ans = 0.0;
    for (int i = 0; i < 2; i++) {
        if (fabs(p[i] - 0.0) < 1e-8) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int X, int Y) ///ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ï¢ï¿½ï¿½
{
    int cnt[4];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (!matrixLAPP[X][j] && !matrixLAPP[Y][j]) {
            cnt[0]++;
        }
        else if (!matrixLAPP[X][j] && matrixLAPP[Y][j]) {
            cnt[1]++;
        }
        else if (matrixLAPP[X][j] && !matrixLAPP[Y][j]) {
            cnt[2]++;
        }
        else if (matrixLAPP[X][j] && matrixLAPP[Y][j]) {
            cnt[3]++;
        }
    }

    double p[4];
    for (int i = 0; i < 4; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = cnt[i] * 1.0 / length;
    }

    double ans = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(p[i] - 0.0) < 1e-6 || p[i] < 0) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int Z, int X, int Y, int chosen) /// H(Z | (X, Y))
{
    int cnt[4];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (!matrixLAPP[Z][j] && !judge(X, Y, chosen, j)){
            cnt[0]++;
        }
        else if (!matrixLAPP[Z][j] && judge(X, Y, chosen, j)){
            cnt[1]++;
        }
        else if (matrixLAPP[Z][j] && !judge(X, Y, chosen, j)){
            cnt[2]++;
        }
        else if (matrixLAPP[Z][j] && judge(X, Y, chosen, j)){
            cnt[3]++;
        }
    }

    double p[4];
    for (int i = 0; i < 4; i++) {
        p[i] = cnt[i] * 1.0 / length;
        //cout << cnt[i] << " ";
    }
    //cout << endl;

    double ans = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(p[i] - 0.0) < 1e-6 || p[i] < 0) continue;
        ans += (p[i] * log(p[i]));
        //printf("%.9lf ", p[i]);
    }
    //cout << endl;
    //cout << "ans:" << ans << endl;
    return -ans;
}

double U(int X, int Y) /// U(X | Y)
{
    double HEX = H(X);
    if (fabs(HEX) < fexp) return 1.0;
    //cout << H(X) << " " << H(Y) << " " << H(X, Y) << endl;
    return (HEX + H(Y) - H(X, Y)) / HEX;
}

double U(int Z, int X, int Y, int chosen) ///Zï¿½Ú£ï¿½X, Y)ï¿½ï¿½ï¿½ï¿½ï¿½ U(Z | (X, Y))
{
    double HEZ = H(Z);
    if (fabs(HEZ) < fexp) return 1.0;
    return (HEZ + H(X, Y, chosen) - H(Z, X, Y, chosen)) / HEZ;
}


bool isremove[1024];

int main()
{
    //freopen("Throat.txt", "r", stdin);
    //freopen("changdao3.txt", "w", stdout);
    //printf("test1");
    freopen("D:\\370v-0-1.txt", "r", stdin);
    //freopen("E:\\v8w.txt", "w", stdout);
    printf("test\n");
    int n;
    scanf("%d", &n);
    //printf("%d test3\n", n);
    char str[MAXN];
    memset(isremove, 0, sizeof(isremove));
    for (int i = 0; i < n; i++) {
        scanf("%s", str);
        //printf("%d line\n", i);
        length = strlen(str);
        int cntone = 0;
        for (int j = 0; j < length; j++) {
        	
            matrixLAPP[i][j] = (str[j] - '0');
            if (matrixLAPP[i][j] == 1) {
                cntone++;
            }
        }
        if (cntone <= 2) isremove[i] = 1;
    }
    printf("%d Íê³ÉÊý¾Ý¶ÁÈ¡\n", n);

	freopen("D:\\high-2-0.2(2).txt", "w", stdout);
    for (int i = 0; i < n; i++) {
        if (isremove[i]) continue;
            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                if (isremove[k]) continue;
                double res1 = U(k, i);
                double res2 = U(i, k);
                if (res1 > 0.2 && res2 > 0.2) {
                // ÕâÀï¿ÉÒÔÌí¼Ó¶îÍâµÄÊä³öÐÅÏ¢
                	printf("Additional Info: V%04d V%04d res1=%.6lf\n", k+1, i+1, res1);
            	}
            }
    }


    return 0;
}
