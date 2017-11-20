#include <iostream>
#include <cstdio>
#include <cstdlib>
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

struct WeakClassifier
{
    int sign;
    int threshold;

    WeakClassifier()
        : sign(1), threshold(0) { }
    WeakClassifier(int _sign, int _threshold)
        : sign(_sign), threshold(_threshold) { }
};

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


Mat get_image(const char *path)
{
    Mat in_img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

    Ptr<CLAHE> clahe_filter = createCLAHE();
    clahe_filter->setClipLimit(2);

    Mat clahed_img;
    clahe_filter->apply(in_img, clahed_img);

    Mat resized_img;
    resize(clahed_img, resized_img, Size(SIZE, SIZE));

    return resized_img;
}

vector<int> get_integral_image(const Mat &img)
{
    vector<int> integral_img;
    integral_img.resize(SIZE * SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int value = 0;

            value = img.at<uchar>(i, j);
            if (i > 0)
                value += integral_img[(i - 1) * SIZE + j];
            if (j > 0)
                value += integral_img[i * SIZE + (j - 1)];
            if (i > 0 && j > 0)
                value -= integral_img[(i - 1) * SIZE + (j - 1)];

            integral_img[i * SIZE + j] = value;
        }
    }

    return integral_img;
}


bool load_images(const char *f_path, vector<char*> &out_path, vector<int> &out_mark, int mark)
{
    FILE *f;
    
    f = fopen(f_path, "r");
    if (f == 0) {
        return 0;
    }

    int p = 0, count = 0;
    char name[100], c;

    while (fscanf(f, "%c", &c) > 0) {
        if (c == '\n') {
            char *s = new char[p + 1];

            memcpy(s, name, p);
            s[p] = '\0';

            out_path.push_back(s);
            out_mark.push_back(mark);

            p = 0;
            count++;
            continue;
        }
        name[p++] = c;
    }

    fclose(f);
    return count;
}

#if 0
// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B) {
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calculate the L2 relative error between images.
        double errorL2 = norm(A, B, CV_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        //Images have a different size
        return 100000000.0;  // Return a bad value
    }
}


void find_face()
{
    const char path_1[] = "C:\\alexmihy\\botalka\\diploma\\databases\\ExtendedYaleB\\yaleB11\\yaleB11_P00A+000E+00.pgm";
    const char path_2[] = "C:\\alexmihy\\botalka\\diploma\\databases\\CroppedYale\\yaleB11\\yaleB11_P00A+000E+00.pgm";

    Mat in_img1 = imread(path_1, CV_LOAD_IMAGE_GRAYSCALE);
    Mat in_img2 = imread(path_2, CV_LOAD_IMAGE_GRAYSCALE);

    Ptr<CLAHE> clahe_filter = createCLAHE();
    clahe_filter->setClipLimit(2);

    Mat img1, img2;
    clahe_filter->apply(in_img1, img1);
    clahe_filter->apply(in_img2, img2);

    imshow("good1", img1);
    imshow("good2", img2);


    for (int i1 = 0; i1 < img1.rows - img2.rows; i1++) {
        for (int j1 = 0; j1 < img1.cols - img2.cols; j1++) {
            Rect myROI(j1, i1, img2.cols, img2.rows);

            Mat cropped = img1(myROI);

            if (getSimilarity(cropped, img2) < 0.3) {
                cout << "GOOD" << endl;
                img1.at<uchar>(i1, j1) = 255;

                imshow("GOT IT", cropped);
                goto ret;

            }

        }

    }
ret:

    imshow("good", img1);
    //imshow("resized", resized_img);
    waitKey();

}

void generate()
{
    const char pathes[] = "C:\\alexmihy\\botalka\\diploma\\databases\\background_samples\\path.txt";
    //const char out_path[] = "C:\\alexmihy\\botalka\\diploma\\databases\\background_samples\\result\\0000.pgm";

    char out_path[100] = { 0 };

    FILE *f = fopen(pathes, "r");

    printf("%.4d\n", 1);

    if (f == 0) {
        cout << "BAD" << endl;
        return 1;
    }

    srand(time(NULL));

    int p = 0, cnt = 0;
    char name[100], c;

    while (fscanf(f, "%c", &c) > 0) {
        if (c == '\n') {
            name[p] = '\0';

            Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);

            for (int i = 0; i < 5; i++) {

                int x = rand() % (img.rows - 100);
                int y = rand() % (img.cols - 100);
                int height = rand() % (img.rows - x - 80) + 80;
                int width = rand() % (img.cols - y - 80) + 80;

                Rect myROI(y, x, width, height);
                Mat cropped = img(myROI);

                out_path[0] = '\0';
                sprintf(out_path, "C:\\alexmihy\\botalka\\diploma\\databases\\background_samples\\result\\%.4d.pgm", cnt);
                cnt++;

                imwrite(out_path, cropped);

                //return 0;

            }
            p = 0;
            continue;
        }
        name[p++] = c;
    }

    return 0;
}
#endif 

int main(int argc, char **argv)
{
    char pos_path[] = "C:\\alexmihy\\botalka\\diploma\\databases\\good_samples\\path.txt";
    char neg_path[] = "C:\\alexmihy\\botalka\\diploma\\databases\\background_samples\\result\\path.txt";

    vector<char*> sample_path;
    vector<int> sample_mark;

    int pos_count, neg_count;

    if (((pos_count = load_images(pos_path, sample_path, sample_mark, +1)) < 0) ||
        ((neg_count = load_images(neg_path, sample_path, sample_mark, -1)) < 0)) {
        cout << "Unable to load image pathes" << endl;
        return 1;
    }

    cout << "PATHES LOADED" << endl;

    // FILL SAMPLE_PATH AND SAMPLE_MARK
    //sample_path.push_back(in_path);
    //sample_mark.push_back(+1);

    int s_count = sample_path.size();
    int f_count = WAVELETS_1X2_COUNT + WAVELETS_2X1_COUNT + 
        WAVELETS_1X3_COUNT + WAVELETS_3X1_COUNT + WAVELETS_2X2_COUNT;
    
    // 2x1 1x2 3x1 1x3 2x2 for each sample
    vector<vector<int>> sample_features;
    sample_features.resize(s_count);

    for (int i = 0; i < s_count; i++) {
        vector<int> feature[5], whole_feature;

        // LOAD IMAGE -> CLAHED -> RESIZED -> INTEGRAL
        Mat img = get_image(sample_path[i]);
        vector<int> integral_img = get_integral_image(img);

        feature[0] = get_2x1_wavelets(integral_img);
        feature[1] = get_1x2_wavelets(integral_img);
        feature[2] = get_3x1_wavelets(integral_img);
        feature[3] = get_1x3_wavelets(integral_img);
        feature[4] = get_2x2_wavelets(integral_img);

        whole_feature.resize(f_count);
        int p = 0;
        for (int j = 0; j < 5; j++) {
            for (size_t k = 0; k < feature[j].size(); k++)
                whole_feature[p++] = feature[j][k];
        }
        sample_features[i] = whole_feature;
    }

    cout << "WAVELETS CALCULATED" << endl;

    vector<WeakClassifier> classifiers;
    classifiers.resize(f_count);
    
    for (int i = 0; i < f_count; i++) {
        vector<pair<int, int>> hist(s_count);
        for (int j = 0; j < s_count; j++) {
            hist[j] = make_pair(sample_features[j][i], sample_mark[j]);
        }

        sort(hist.begin(), hist.end());
        int plus_sum = 0, minus_sum = 0;
        double max = -1, min = -1;
        int max_idx, min_idx;


        for (int j = 0; j < s_count; j++) {
            if (hist[j].second == 1)
                plus_sum++;
            else
                minus_sum++;

            if (j - 1 >= 0 && hist[j].first == hist[j - 1].first)
                continue;

            double val = 1.0 * (plus_sum + (minus_count - minus_sum)) / s_count;

            if (max < 0 || max < val) {
                max = val;
                max_idx = j;
            }
            if (min < 0 || min > val) {
                min = val;
                min_idx = j;
            }
        }

        if (1 - min > max) {
            //направо строго
            classifiers[i].sign = -1;
            classifiers[i].threshold = hist[min_idx].first;

            if (fabs(min) < 1e-4) {
                classifiers[i].threshold = rand() % 255;
                cout << "100%" << endl;
            }
        } else {
            //налево нестрого
            classifiers[i].sign = 1;
            classifiers[i].threshold = hist[max_idx].first;

            if (fabs(max - 1) < 1e-4) {
                classifiers[i].threshold = rand() % 255;
                cout << "100%" << endl;
            }
        }
    }

    cout << "WEAK DONE" << endl;

    f = fopen("C:\\alexmihy\\botalka\\diploma\\code\\alpha.txt", "w");

    if (f == 0) {
        cout << "Can't open out" << endl;
        return 1;
    }

    // P >= V по дефолту
    vector<double> weights(s_count);
    for (int i = 0; i < s_count; i++)
        weights[i] = 1.0 / s_count;

    vector<pair<int, double>> alphas;
    for (int K = 0; K < 1000; K++) {
        int min_idx;
        double min_error = -1;
        for (int i = 0; i < f_count; i++) {
            double error = 0;
            for (int j = 0; j < s_count; j++) {
                int res;
                if (classifiers[i].sign > 0)
                    res = classifiers[i].threshold >= sample_features[j][i] ? 1 : -1;
                else 
                    res = classifiers[i].threshold < sample_features[j][i] ? 1 : -1;

                if (sample_mark[j] != res)
                    error += weights[j];
            }

            if (min_error < 0 || min_error > error) {
                min_error = error;
                min_idx = i;
            }
        }


        double alpha = log((1 - min_error) / min_error) / 2;
        alphas.push_back(make_pair(min_idx, alpha));

        for (int i = 0; i < s_count; i++) {
            int res;
            if (classifiers[min_idx].sign > 0)
                res = classifiers[min_idx].threshold >= sample_features[i][min_idx] ? 1 : -1;
            else
                res = classifiers[min_idx].threshold < sample_features[i][min_idx] ? 1 : -1;

            if (sample_mark[i] != res)
                weights[i] *= 0.5 / min_error;
            else
                weights[i] *= 0.5 / (1 - min_error);
            
        }
    }

    /*
    cout << "ALPHAS:" << endl;
    for (size_t i = 0; i < alphas.size(); i++) {
        cout << alphas[i].first << " - " << alphas[i].second << endl;
    }
    */
    for (size_t i = 0; i < alphas.size(); i++)
        fprintf(f, "%d %lf\n", alphas[i].first, alphas[i].second);



    vector<char*> test_path;
    vector<int> test_mark;


    s_count = test_path.size();
    int correct = 0;
    for (int i = 0; i < s_count; i++) {
        vector<int> feature[5], whole_feature;

        // LOAD IMAGE -> CLAHED -> RESIZED -> INTEGRAL
        Mat img = get_image(test_path[i]);
        vector<int> integral_img = get_integral_image(img);

        feature[0] = get_2x1_wavelets(integral_img);
        feature[1] = get_1x2_wavelets(integral_img);
        feature[2] = get_3x1_wavelets(integral_img);
        feature[3] = get_1x3_wavelets(integral_img);
        feature[4] = get_2x2_wavelets(integral_img);

        whole_feature.resize(f_count);
        int p = 0;
        for (int j = 0; j < 5; j++) {
            for (size_t k = 0; k < feature[j].size(); k++)
                whole_feature[p++] = feature[j][k];
        }

        double result = 0;
        for (size_t k = 0; k < alphas.size(); k++) {
            int res;
            int idx = alphas[k].first;

            if (classifiers[idx].sign > 0)
                res = classifiers[idx].threshold >= whole_feature[idx] ? 1 : -1;
            else
                res = classifiers[idx].threshold < whole_feature[idx] ? 1 : -1;

            result += alphas[k].second * res;
        }

        if (result * test_mark[i] > 0)
            correct++;
    }

    cout << 1.0 * correct / s_count << endl;

    return 0;
}
