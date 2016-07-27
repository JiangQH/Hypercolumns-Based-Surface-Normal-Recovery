/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: 2016年06月06日 星期一 20时33分55秒
 ************************************************************************/

#include <iostream>
#include <vector>
#include <map>
#include <cmath>

using namespace std;

void get_map_point(std::map<int, double>& result,
        const vector<int>& coordinates, const vector<int>& original_size,
        double scale) {
    /***
     * This function uses the bilinear interpolation to locate the corresponding
     * point in the original feature map with their corresponding weights.
     * note, since the orinal size is known, represents the coordinates by a single
     * value
     * */
   int x = coordinates[0];
   int y = coordinates[1];
   int w = original_size[0];
   int h = original_size[1];
   double r = x / scale + 1.0 / (2.0 * scale) - 0.5;
   double c = y / scale + 1.0 / (2.0 * scale) - 0.5;
   int u = floor(r);
   int v = floor(c);
   double delta_r = r - u;
   double delta_c = c - v;
   if (u < 0)
       delta_r = 1;
   if (u + 1 >= h)
       delta_r = 0;
   if (v < 0)
       delta_c = 1;
   if (v + 1 >= w)
       delta_c = 0;
   result.clear();
   if ((1-delta_r) * (1-delta_c) != 0)
       result.insert(std::make_pair(u * w + v, 
                   (1-delta_r)* (1-delta_c)));
   if (delta_r * (1-delta_c) != 0)
       result.insert(std::make_pair((u+1)*w + v,
                   delta_r * (1-delta_c)));
   if (delta_c * (1-delta_r) != 0)
       result.insert(std::make_pair(u * w + v + 1,
                   delta_c * (1-delta_r)));
   if (delta_r * delta_c != 0)
       result.insert(std::make_pair((u+1)*w + v + 1,
                   delta_r * delta_c));

}

int main() {
    map<int, double> result;
    vector<int> coordinates;
    vector<int> original_size;
    double scale = 2;
    coordinates.push_back(0);
    coordinates.push_back(0);
    original_size.push_back(2);
    original_size.push_back(2);
    get_map_point(result, coordinates,
            original_size, scale);
    for (auto it = result.cbegin(); it != result.cend(); ++it) {
        cout << (*it).first << ":" << (*it).second << endl;
    }
    return 0;
}
