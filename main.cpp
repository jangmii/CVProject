#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>
#define MAXNUM 30
#define LEVEL 2 // <-------------------- ���⼭ ���� ����
using namespace std;
using namespace cv;
using namespace cv::dnn;

string protoFile = "hand/pose_deploy.prototxt";
string weightsFile = "hand/pose_iter_102000.caffemodel";
string imageFiles[3] = {"�δ���.png", "��״�.png", "����.png"};
string typeName[3] = { "�δ���","Ư���� �δ���","����" };

enum State {
    STATE_BEFORE, STATE_NOW, STATE_AFTER
};

const int timeLimit = 32;
const int type_prop[10] = { 0,0,0,0,0,1,1,1,2,2 }; //������ Ȯ��
const int point[3] = { +1, +3, -2 }; //�δ��� ������ ����Ʈ
const int lv_num[3] = { 10, 15, 20 };
const int lv_duration[3] = {3,2,1};

int num = 15; //�δ��� ����
double duration = 2; //�δ��� ���ӽð�

int main(int argc, char** argv)
{
    float thresh = 0.01;
    num = lv_num[LEVEL -1];
    duration = lv_duration[LEVEL -1];

    std::cout << "ī�޶� ���� ��...\n";
    VideoCapture cap(1, CAP_DSHOW);
    if (!cap.isOpened()) cap = VideoCapture(0, CAP_DSHOW);
    if (!cap.isOpened()) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
    
    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    float aspect_ratio = frameWidth / (float)frameHeight;
    int inHeight = 368;
    int inWidth = (int(aspect_ratio * inHeight) * 8) / 8;
    std::cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;
    
    std::cout << "�� �д� ��...\n";
    Net net = readNet(protoFile, weightsFile);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    double start;
    int score = 0;
    Point pos[MAXNUM] = {}; //�δ��� ��ġ
    int state[MAXNUM] = {}; //�δ��� ����: STATE_BEFORE(���� ��), STATE_NOW(���� ��), STATE_AFTER(�����-�ð��ʰ� or ���� ��)
    int type[MAXNUM] = {};  //�δ��� ����: 0(�Ϲݵδ���), 1(��״�), 2(����)
    int stime[MAXNUM] = {}; //�δ��� ���� �ð�
   
    //�̹��� �ҷ�����
    std::cout << "�δ��� ���� ��...\n";
    Mat img[3], img_gray[3];
    uniform_int_distribution<int> dis[3][2], disType(0, 9), disTime(0, timeLimit - (duration+0.5)/1);
    for (int i = 0; i < 3; i++) {
        img[i] = imread(imageFiles[i]);
        resize(img[i], img[i], Size(0, 0), 0.3, 0.3);
        cvtColor(img[i], img_gray[i], COLOR_BGR2GRAY);
        dis[i][0] = uniform_int_distribution<int>(0, inHeight - img[i].cols);
        dis[i][1] = uniform_int_distribution<int>(0, inWidth - img[i].rows);
    }
    
    //����(�δ��� ��ġ, ����) ����
    std::cout << "�δ��� �̵� ��...\n\n";
    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < num; i++) {
        type[i] = type_prop[disType(gen)];
        pos[i] = Point(dis[type[i]][0](gen), dis[type[i]][0](gen));
        state[i] = STATE_BEFORE;
        stime[i] = disTime(gen);
    }
    while (1) { //���ȭ��
        cap >> frame;
        flip(frame, frame, 1);
        cv::putText(frame, "Press any key for start...", cv::Point(170, 450), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
        cv::imshow("�δ����� ����", frame);
        if (waitKey(1) != -1) break;
    }
    start = (double)cv::getTickCount();
    while (1)
    {
        double t = (double)cv::getTickCount();
        double timePassed = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
        for (int i = 0; i < num; i++) {
            if (state[i] == STATE_BEFORE) {
                if (timePassed - stime[i] >= 0 && timePassed - stime[i] <= duration)
                    state[i] = STATE_NOW;
            }
            else if (state[i] == STATE_NOW) {
                if (timePassed - stime[i] > duration)
                    state[i] = STATE_AFTER;
            }
        }
        cap >> frame;
        cv::flip(frame, frame, 1);
        Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
        net.setInput(inpBlob);
        Mat output = net.forward();
        Rect rect[MAXNUM];

        //�δ��� �ٿ��ֱ�
        for (int i = 0; i < num; i++) {
            if (state[i] == STATE_NOW) {
                rect[i] = Rect(pos[i], img[type[i]].size());
                //std::cout << img[type[i]].size() << endl;
                Mat roi = frame(rect[i]), mask = img_gray[type[i]];
                img[type[i]].copyTo(roi, mask);
            }
        }
        
        int H = output.size[2];
        int W = output.size[3];
        Mat probMap(H, W, CV_32F, output.ptr(0, 8)); //8:���� ��
        resize(probMap, probMap, Size(frameWidth, frameHeight));

        Point maxLoc;
        double prob;
        cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        if (prob > thresh){
            Point now((int)maxLoc.x, (int)maxLoc.y);
            circle(frame, now, 8, Scalar(0, 255, 255), -1);
            for (int i = 0; i < num; i++) {
                if (state[i] == STATE_NOW) {
                    if (rect[i].contains(now)) {
                        state[i] = STATE_AFTER;
                        score += point[type[i]];
                        cv::putText(frame, cv::format("%+d", point[type[i]]), pos[i], cv::FONT_HERSHEY_SIMPLEX, 1.0, type[i]<=1?cv::Scalar(50, 250, 50): cv::Scalar(50, 50, 250), 3);
                        std::cout << typeName[type[i]] << "�� ��Ҵ�! : "<< point[type[i]] << "��\n";
                    }
                }
            }
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        timePassed = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
        if (timePassed > timeLimit) {
            cv::putText(frame, cv::format("Final score: %d ", score), cv::Point(150, 130), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 100, 0), 3);
            cv::putText(frame, "Press any key for quit...", cv::Point(170, 450), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(250, 250, 250), 1);
            cv::imshow("�δ����� ����", frame);
            if(waitKey()) break;
        }
        //std::cout << "fps = " << 1/t << endl;
        cv::putText(frame, cv::format("score: %d ", score), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 0), 2);
        cv::putText(frame, cv::format("time left: %.2f sec", timeLimit - timePassed), cv::Point(350, 50), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(0, 0, 200), 2);
        cv::imshow("�δ����� ����", frame);
        char key = waitKey(1);
        if (key == 27)
            break;  
    }
    cout << "                                   +\n";
    cout << "---------------------------------------\n";
    cout << "\n����: " << score << endl;
    cap.release();
    return 0;
}
