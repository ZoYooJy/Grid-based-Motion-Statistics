#include "myGMS.h"

inline cv::Mat Drawer(cv::Mat& img1, vector<cv::KeyPoint>& vkps1, cv::Mat& img2,
                      vector<cv::KeyPoint>& vkps2, vector<cv::DMatch>& matches,
                      int type) {
  const int height = max(img1.rows, img2.rows);
  const int width = img1.cols + img2.cols;
  cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
  img2.copyTo(output(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

  if (type == 1) {
    for (size_t i = 0; i < matches.size(); i++) {
      cv::Point2f left = vkps1[matches[i].queryIdx].pt;
      cv::Point2f right =
          (vkps2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));
      line(output, left, right, cv::Scalar(0, 255, 255));
    }
  } else if (type == 2) {
    for (size_t i = 0; i < matches.size(); i++) {
      cv::Point2f left = vkps1[matches[i].queryIdx].pt;
      cv::Point2f right =
          (vkps2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));
      line(output, left, right, cv::Scalar(255, 0, 0));
    }

    for (size_t i = 0; i < matches.size(); i++) {
      cv::Point2f left = vkps1[matches[i].queryIdx].pt;
      cv::Point2f right =
          (vkps2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));
      circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
      circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
    }
  }

  return output;
}

inline void GMSMatch(cv::Mat& img1, cv::Mat& img2) {
  vector<cv::KeyPoint> vkps1, vkps2;
  cv::Mat desc1, desc2;
  vector<cv::DMatch> matchesAll, matchesGMS;

  cv::Ptr<cv::ORB> orb = cv::ORB::create(10000);
  orb->setFastThreshold(0);
  orb->detectAndCompute(img1, cv::Mat(), vkps1, desc1);
  orb->detectAndCompute(img2, cv::Mat(), vkps2, desc2);

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(desc1, desc2, matchesAll);
  cv::Mat res_orig = Drawer(img1, vkps1, img2, vkps2, matchesAll, 1);
  cv::imshow("original matches", res_orig);
  // cv::waitKey();

  vector<bool> vbInliers;
  gms_matcher myGMS(vkps1, img1.size(), vkps2, img2.size(), matchesAll);
  int numInliers = myGMS.getInlierMask(vbInliers, false, false);
  cout << "Get total " << numInliers << " matches." << endl;

  // get inliers
  for (size_t i = 0; i < numInliers; i++)
    if (vbInliers[i]) matchesGMS.push_back(matchesAll[i]);

  //* sort the matches according to the descriptor distance
  sort(matchesGMS.begin(), matchesGMS.end(),
       [](const cv::DMatch& a, const cv::DMatch& b) {
         return a.distance < b.distance;
       });

  /*find the relationship between number of kps and calculate Homography cost*/
  vector<int> queryIdx(matchesGMS.size()), trainIdx(matchesGMS.size());
  for (size_t i = 0; i < matchesGMS.size(); i++) {
    queryIdx[i] = matchesGMS[i].queryIdx;
    trainIdx[i] = matchesGMS[i].trainIdx;
  }
  vector<cv::Point2f> pts1, pts2;
  cv::KeyPoint::convert(vkps1, pts1, queryIdx);
  cv::KeyPoint::convert(vkps2, pts2, trainIdx);

  vector<pair<int, double>> f_num_time(
      matchesGMS.size());  //<nums of kps, Cal H cost>
  vector<cv::Point2f> sub1, sub2;
  for (int iter = 100; iter < 500; iter++) {
    sub1.clear();
    sub2.clear();
    sub1.assign(pts1.begin(), pts1.begin() + iter);
    sub2.assign(pts2.begin(), pts2.begin() + iter);

    double sTime = (double)cv::getTickCount();
    cv::Mat H = findHomography((cv::Mat)sub1, (cv::Mat)sub2, cv::RHO, 3);
    double interval =
        ((double)cv::getTickCount() - sTime) / cv::getTickFrequency();

    f_num_time.push_back(make_pair(iter, interval));
  }
  // for (auto& p : f_num_time) cout << p.first << " " << p.second << endl;
  /*find the relationship between number of kps and calculate Homography cost*/

  int len = 300;
  vector<cv::DMatch> dm(len);
  dm.assign(matchesGMS.begin(), matchesGMS.begin() + len);
  cv::Mat res = Drawer(img1, vkps1, img2, vkps2, dm, 1);
  cv::imshow("after GMS filter", res);
  cv::waitKey();
}

int main(int argc, char** argv) {
  cv::Mat img1 = cv::imread("../000000.png");
  cv::Mat img2 = cv::imread("../000001.png");
  // cv::Mat img1_, img2_, _img1_, _img2_;
  // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  // clahe->apply(img1, img1_);
  // clahe->apply(img2, img2_);
  // cv::cvtColor(img1_, _img1_, cv::COLOR_GRAY2BGR);
  // cv::cvtColor(img2_, _img2_, cv::COLOR_GRAY2BGR);
  // cv::imshow("img1_clahe", _img1_);
  // cv::imshow("img2_clahe", _img2_);
  // GMSMatch(_img1_, _img2_);
  GMSMatch(img1, img2);
  return 0;
}
