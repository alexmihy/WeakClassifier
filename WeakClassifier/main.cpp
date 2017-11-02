#include <iostream>
#include <cstdio>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define SIZE 24

#define WAVELETS_1X2_COUNT 82800
#define WAVELETS_2X1_COUNT 82800
#define WAVELETS_1X3_COUNT 75900
#define WAVELETS_3X1_COUNT 75900
#define WAVELETS_2X2_COUNT 76176

int get_value(vector<int> &mask, int row1, int col1, int row2, int col2)
{
    int ret;
    
    ret = mask[row1 * SIZE + col1];
    if (row2 > 0)
        ret -= mask[(row2 - 1) * SIZE + col1];
    if (col2 > 0)
        ret -= mask[row1 * SIZE + (col2 - 1)];
    if (row2 > 0 && col2 > 0)
        ret += mask[(row2 - 1) * SIZE + (col2 - 1)];

    return ret;
}

vector<int> get_2x1_wavelets(vector<int> &mask)
{
    int p = 0;
    vector<int> ret;
    ret.resize(WAVELETS_2X1_COUNT);

    for (char i1 = 1; i1 < SIZE; i1++) {
        for (char j1 = 0; j1 < SIZE; j1++) {
            
            for (char i2 = 0; i2 < i1; i2++) {
                for (char j2 = 0; j2 <= j1; j2++) {
                    char i_mid = (i1 + i2) / 2;
                    ret[p] = -get_value(mask, i1, j1, i_mid + 1, j2);
                    ret[p] += get_value(mask, i_mid, j1, i2, j2);
                    p++;
                }
            }
            
        }
    }

    return ret;
}

vector<int> get_1x2_wavelets(vector<int> &mask)
{
    int p = 0;
    vector<int> ret;
    ret.resize(WAVELETS_1X2_COUNT);

    for (char i1 = 0; i1 < SIZE; i1++) {
        for (char j1 = 1; j1 < SIZE; j1++) {

            for (char i2 = 0; i2 <= i1; i2++) {
                for (char j2 = 0; j2 < j1; j2++) {
                    char j_mid = (j1 + j2) / 2;
                    ret[p] = -get_value(mask, i1, j1, i2, j_mid + 1);
                    ret[p] += get_value(mask, i1, j_mid, i2, j2);
                    p++;
                }
            }

        }
    }

    return ret;
}

vector<int> get_1x3_wavelets(vector<int> &mask)
{
    int p = 0;
    vector<int> ret;
    ret.resize(WAVELETS_1X3_COUNT);

    for (char i1 = 0; i1 < SIZE; i1++) {
        for (char j1 = 2; j1 < SIZE; j1++) {

            for (char i2 = 0; i2 <= i1; i2++) {
                for (char j2 = 0; j2 < j1 - 1; j2++) {
                    char j_13 = j2 + (j1 - j2) / 3;
                    char j_23 = j2 + (2 * (j1 - j2)) / 3;
                    ret[p] = -get_value(mask, i1, j_23, i2, j_13 + 1);
                    ret[p] += get_value(mask, i1, j1, i2, j_23 + 1);
                    ret[p] += get_value(mask, i1, j_13, i2, j2);
                    p++;
                }
            }

        }
    }

    return ret;
}

vector<int> get_3x1_wavelets(vector<int> &mask)
{
    int p = 0;
    vector<int> ret;
    ret.resize(WAVELETS_3X1_COUNT);

    for (char i1 = 2; i1 < SIZE; i1++) {
        for (char j1 = 0; j1 < SIZE; j1++) {

            for (char i2 = 0; i2 < i1 - 1; i2++) {
                for (char j2 = 0; j2 <= j1; j2++) {
                    char i_13 = i2 + (i1 - i2) / 3;
                    char i_23 = i2 + (2 * (i1 - i2)) / 3;
                    ret[p] = -get_value(mask, i_23, j1, i_13 + 1, j2);
                    ret[p] += get_value(mask, i1, j1, i_23 + 1, j2);
                    ret[p] += get_value(mask, i_13, j1, i2, j2);
                    p++;
                }
            }

        }
    }

    return ret;
}

vector<int> get_2x2_wavelets(vector<int> &mask)
{
    int p = 0;
    vector<int> ret;
    ret.resize(WAVELETS_2X2_COUNT);

    for (char i1 = 1; i1 < SIZE; i1++) {
        for (char j1 = 1; j1 < SIZE; j1++) {

            for (char i2 = 0; i2 < i1; i2++) {
                for (char j2 = 0; j2 < j1; j2++) {
                    char i_mid = (i1 + i2) / 2;
                    char j_mid = (j1 + j2) / 2;
                    ret[p] = -get_value(mask, i_mid, j1, i2, j_mid + 1);
                    ret[p] -= get_value(mask, i1, j_mid, i_mid + 1, j2);
                    ret[p] += get_value(mask, i1, j1, i_mid + 1, j_mid + 1);
                    ret[p] += get_value(mask, i_mid, j_mid, i2, j2);
                    p++;
                }
            }

        }
    }

    return ret;
}


int main(int argc, char **argv)
{
    char in_path[] = "C:\\alexmihy\\botalka\\diploma\\code\\image.pgm";
    char out_path[] = "C:\\alexmihy\\botalka\\diploma\\code\\out.bmp";

    Mat in_img = imread(in_path, CV_LOAD_IMAGE_GRAYSCALE);
    //imshow("input", in_img);

    Ptr<CLAHE> clahe_filter = createCLAHE();
    clahe_filter->setClipLimit(2);

    Mat clahed_img;
    clahe_filter->apply(in_img, clahed_img);
    //imshow("clahed", clahed_img);

    Mat resized_img;
    resize(clahed_img, resized_img, Size(SIZE, SIZE));
    //imshow("resized", resized_img);

    vector<int> integral_img;
    integral_img.resize(SIZE * SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int value = 0;

            value = resized_img.at<uchar>(i, j);
            if (i > 0)
                value += integral_img[(i - 1) * SIZE + j];
            if (j > 0)
                value += integral_img[i * SIZE + (j - 1)];
            if (i > 0 && j > 0)
                value -= integral_img[(i - 1) * SIZE + (j - 1)];

            integral_img[i * SIZE + j] = value;
        }
    }

    vector<int> w2x1 = get_2x1_wavelets(integral_img);
    vector<int> w1x2 = get_1x2_wavelets(integral_img);
    vector<int> w1x3 = get_1x3_wavelets(integral_img);
    vector<int> w3x1 = get_3x1_wavelets(integral_img);
    vector<int> w2x2 = get_2x2_wavelets(integral_img);

    printf("%zu\n", w2x1.size());
    printf("%zu\n", w1x2.size());
    printf("%zu\n", w1x3.size());
    printf("%zu\n", w3x1.size());
    printf("%zu\n", w2x2.size());








    //waitKey();

    return 0;
}