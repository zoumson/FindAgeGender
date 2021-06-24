#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>

std::tuple<cv::Mat, std::vector<std::vector<int>>> getFaceBox(cv::dnn::Net net, cv::Mat &frame, double conf_threshold)
{
    cv::Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    cv::Size size = cv::Size(300, 300);
    // std::std::vector<int> meanVal = {104, 117, 123};
    cv::Scalar meanVal = cv::Scalar(104, 117, 123);

    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    std::vector<std::vector<int>> bboxes;

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            std::vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}

int main(int argc, char** argv)
{

   cv::String keys =
        "{i input |<none>           | image or video path}"                           
        "{e fproto |./resource/proto/opencv_face_detector.pbtxt           | face proto}"                                                   
        "{f fmodel |./resource/model/opencv_face_detector_uint8.pb           | face model}"  
        "{a aproto |./resource/proto/age_deploy.prototxt           | age proto}"   
        "{b amodel |./resource/model/age_net.caffemodel          | age model}" 
        "{g gproto |./resource/proto/gender_deploy.prototxt           | gender proto}"   
        "{h gmodel |./resource/model/gender_net.caffemodel         | gender model}"                                                                                                            
        "{s save |       | save image with age and gender}"                                                                                                            
        "{help h usage ?    |      | show help message}";      
  
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Face Age Gender Detection");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }

    cv::String input = parser.get<cv::String>("input"); 
    cv::String faceProto = parser.get<cv::String>("fproto"); 
    cv::String faceModel = parser.get<cv::String>("fmodel"); 
    cv::String ageProto = parser.get<cv::String>("aproto"); 
    cv::String ageModel = parser.get<cv::String>("amodel"); 
    cv::String genderProto = parser.get<cv::String>("gproto"); 
    cv::String genderModel = parser.get<cv::String>("gmodel"); 
    cv::String save;
    if (parser.has("save")) save = parser.get<cv::String>("save"); 


 
    if (!parser.check()) 
    {
        parser.printErrors();
        return -1;
    }

    cv::Scalar MODEL_MEAN_VALUES = cv::Scalar(78.4263377603, 87.7689143744, 114.895847746);

    std::vector<std::string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
      "(38-43)", "(48-53)", "(60-100)"};

    std::vector<std::string> genderList = {"Male", "Female"};


    // Load Network
    cv::dnn::Net ageNet = cv::dnn::readNet(ageModel, ageProto);
    cv::dnn::Net genderNet = cv::dnn::readNet(genderModel, genderProto);
    cv::dnn::Net faceNet = cv::dnn::readNet(faceModel, faceProto);

        std::cout << "Using CPU device\n";
        ageNet.setPreferableBackend( cv::dnn::DNN_TARGET_CPU);
        
        genderNet.setPreferableBackend( cv::dnn::DNN_TARGET_CPU);

        faceNet.setPreferableBackend( cv::dnn::DNN_TARGET_CPU);

    int padding = 0;
    cv::VideoCapture cap;
    cap.open(input);
    while(cv::waitKey(1) < 0) 
    {
      // read frame
      cv::Mat frame;
      cap.read(frame);
      if (frame.empty())
      {
          cv::waitKey();
          break;
      }

      std::vector<std::vector<int>> bboxes;
      cv::Mat frameFace;
      std::tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);

      if(bboxes.size() == 0) 
      {
        std::cout << "No face detected, checking next frame.\n";
        continue;
      }
      for (auto it = begin(bboxes); it != end(bboxes); ++it) 
      {
        cv::Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
        cv::Mat face = frame(rec); // take the ROI of box on the frame
        // Use the face as gender network input
        cv::Mat blob;
        blob = cv::dnn::blobFromImage(face, 1, cv::Size(227, 227), MODEL_MEAN_VALUES, false);
        genderNet.setInput(blob);
        // std::string gender_preds;
        std::vector<float> genderPreds = genderNet.forward();
        // printing gender here
        // find max element index
        // distance function does the argmax() work in C++
        int max_index_gender = std::distance(genderPreds.begin(), std::max_element(genderPreds.begin(), genderPreds.end()));
        std::string gender = genderList[max_index_gender];
        std::cout << "Gender: " << gender << "\n";

        /* // Uncomment if you want to iterate through the gender_preds std::vector
        for(auto it=begin(gender_preds); it != end(gender_preds); ++it) {
          std::cout << *it << endl;
        }
        */

        ageNet.setInput(blob);
        std::vector<float> agePreds = ageNet.forward();
        /* // uncomment below code if you want to iterate through the age_preds
         * std::vector
        std::cout << "PRINTING AGE_PREDS" << endl;
        for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
          std::cout << *it << endl;
        }
        */

        // finding maximum indicd in the age_preds std::vector
        int max_indice_age = std::distance(agePreds.begin(), std::max_element(agePreds.begin(), agePreds.end()));
        std::string age = ageList[max_indice_age];
        std::cout << "Age: " << age << "\n";
        std::string label = gender + ", " + age; // label
        cv::putText(frameFace, label, cv::Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::imshow("Frame", frameFace);
        if (parser.has("save")) cv::imwrite(save,frameFace);
      }

    }
}
