#ifndef MYGMS_H
#define MYGMS_H

#include <ctime>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

#define THRES_FACTOR 6

class gms_matcher {
 public:
  /**
   * @brief Construct a new gms matcher object
   *
   * @param vkps1
   * @param size1
   * @param vkps2
   * @param size2
   * @param vDMatches
   */
  gms_matcher(const vector<cv::KeyPoint>& vkps1, const cv::Size size1,
              const vector<cv::KeyPoint>& vkps2, const cv::Size size2,
              const vector<cv::DMatch>& vDMatches) {
    normalizePts(vkps1, size1, mvp1);
    normalizePts(vkps2, size2, mvp2);
    mNumMatches = vDMatches.size();
    convert(vDMatches, mvMatches);

    mGridSizeLeft = cv::Size(20, 20);
    mPixelNumPerGridLeft = mGridSizeLeft.width * mGridSizeLeft.height;
    mGridNeighborLeft = cv::Mat::zeros(mPixelNumPerGridLeft, 9, CV_32SC1);
    initNB(mGridNeighborLeft, mGridSizeLeft);
  }

  ~gms_matcher() {}

  /**
   * @brief Get the Inlier Mask object
   *
   * @param vbInliers
   * @param withScale
   * @param withRotation
   * @return max number of inliers
   */
  //! outlier interface
  int getInlierMask(vector<bool>& vbInliers, bool withScale = false,
                    bool withRotation = false) {
    int maxInliers = 0;

    if (!withScale && !withRotation) {  //无尺度变换 & 无旋转
      setScale(0);
      // cout << "end setScale" << endl;
      maxInliers = run(1);
      // cout << "end run" << endl;
      vbInliers = mvInlierMask;
      return maxInliers;
    }

    if (withRotation && withScale) {  //有尺度变换 & 有旋转
      for (int Scale = 0; Scale < 5; Scale++) {
        setScale(Scale);
        for (int RotationType = 1; RotationType <= 8; RotationType++) {
          int num_inlier = run(RotationType);

          if (num_inlier > maxInliers) {
            vbInliers = mvInlierMask;
            maxInliers = num_inlier;
          }
        }
      }
      return maxInliers;
    }

    if (withRotation && !withScale) {  //无尺度变换 & 有旋转
      setScale(0);
      for (int RotationType = 1; RotationType <= 8; RotationType++) {
        int num_inlier = run(RotationType);

        if (num_inlier > maxInliers) {
          vbInliers = mvInlierMask;
          maxInliers = num_inlier;
        }
      }
      return maxInliers;
    }

    if (!withRotation && withScale) {  //有尺度变换 & 无旋转
      for (int Scale = 0; Scale < 5; Scale++) {
        setScale(Scale);

        int num_inlier = run(1);

        if (num_inlier > maxInliers) {
          vbInliers = mvInlierMask;
          maxInliers = num_inlier;
        }
      }
      return maxInliers;
    }
  }

 private:
  /**
   * @brief normalize cv::KeyPoint to cv::Point2f
   *
   * @param vkps
   * @param size
   * @param pts
   */
  void normalizePts(const vector<cv::KeyPoint>& vkps, const cv::Size& size,
                    vector<cv::Point2f>& pts) {
    const size_t cnt = vkps.size();
    const int width = size.width;
    const int height = size.height;
    pts.resize(cnt);

    for (size_t i = 0; i < cnt; i++) {
      pts[i].x = vkps[i].pt.x / width;
      pts[i].y = vkps[i].pt.y / height;
      //   todo: may should be
      // pts[i].x = vkps[i].pt.x / height;
      // pts[i].y = vkps[i].pt.y / width;
    }
  }

  /**
   * @brief convert cv::DMatch to vector<pair<ix, idx>>
   *
   * @param vDMatches
   * @param vMatches
   */
  void convert(const vector<cv::DMatch>& vDMatches,
               vector<pair<int, int>>& vMatches) {
    vMatches.resize(mNumMatches);
    for (size_t i = 0; i < mNumMatches; i++)
      vMatches[i] =
          pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
  }

  // TODO
  int getGridIdxLeft(const cv::Point2f& pt, int type) {
    int x = 0, y = 0;

    // floor()：向下取整
    if (type == 1) {
      x = floor(pt.x * mGridSizeLeft.width);
      y = floor(pt.y * mGridSizeLeft.height);

      if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width) {
        return -1;
      }
    }

    if (type == 2) {
      x = floor(pt.x * mGridSizeLeft.width + 0.5);
      y = floor(pt.y * mGridSizeLeft.height);

      if (x >= mGridSizeLeft.width || x < 1) {
        return -1;
      }
    }

    if (type == 3) {
      x = floor(pt.x * mGridSizeLeft.width);
      y = floor(pt.y * mGridSizeLeft.height + 0.5);

      if (y >= mGridSizeLeft.height || y < 1) {
        return -1;
      }
    }

    if (type == 4) {
      x = floor(pt.x * mGridSizeLeft.width + 0.5);
      y = floor(pt.y * mGridSizeLeft.height + 0.5);

      if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width ||
          x < 1) {
        return -1;
      }
    }

    return x + y * mGridSizeLeft.width;
  }

  // TODO
  int getGridIdxRight(const cv::Point2f& pt) {
    int x = floor(pt.x * mGridSizeRight.width);
    int y = floor(pt.y * mGridSizeRight.height);

    return x + y * mGridSizeRight.width;
  }

  /**
   * @brief set the scale in right image
   *
   * @param scale
   */
  void setScale(int scale) {
    mGridSizeRight.width = mGridSizeLeft.width * mScaleRatios[scale];
    mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[scale];

    mPixelNumPerGridRight = mGridSizeRight.width * mGridSizeRight.height;

    mGridNeighborRight = cv::Mat::zeros(mPixelNumPerGridRight, 9, CV_32SC1);
    initNB(mGridNeighborRight, mGridSizeRight);
  }

  /**
   * @brief execute in the constructor
   *
   * @param [out] neighbor 400*9
   * @param [in]  gridSize single cell size: 20*20
   */
  void initNB(cv::Mat& neighbor, const cv::Size& gridSize) {
    for (int i = 0; i < neighbor.rows; i++) {
      vector<int> NB9 = getNB9(i, gridSize);
      int* data = neighbor.ptr<int>(i);  // pixel value
      memcpy(data, &NB9[0], sizeof(int) * 9);
    }
  }

  /**
   * @brief 获取邻域内网格索引
   *
   * @param idx 指定像素索引
   * @param gridSize 单个网格尺寸20*20
   * @return vector<int> 指定像素的邻域网格索引集
   */
  vector<int> getNB9(const int idx, const cv::Size& gridSize) {
    vector<int> NB9(9, -1);

    // grid idx
    int xIdx = idx % gridSize.width;
    int yIdx = idx / gridSize.height;

    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int xIdx_ = xIdx + x;
        int yIdx_ = yIdx + y;

        if (xIdx_ < 0 || xIdx_ >= gridSize.width || yIdx_ < 0 ||
            yIdx_ >= gridSize.height)
          continue;

        NB9[x + 4 + y * 3] = xIdx_ + yIdx_ * gridSize.width;
      }
    }

    return NB9;
  }

  /**
   * @brief 根据网格类型更新mMotionStatistics & mNumKpsPerCellLeft
   *
   * @param gridType
   */
  void assignMatchPairs(int gridType) {
    for (size_t i = 0; i < mNumMatches; i++) {
      cv::Point2f& lp = mvp1[mvMatches[i].first];
      cv::Point2f& rp = mvp2[mvMatches[i].second];

      // get kps's idx in left & right image-grid
      int lgIdx = mvMatchPairs[i].first = getGridIdxLeft(lp, gridType);
      int rgIdx = -1;

      if (gridType == 1)
        rgIdx = mvMatchPairs[i].second = getGridIdxRight(rp);
      else
        rgIdx = mvMatchPairs[i].second;

      if (lgIdx < 0 || rgIdx < 0) continue;

      mMotionStatistics.at<int>(lgIdx, rgIdx)++;
      mNumKpsPerCellLeft[lgIdx]++;
    }
  }

  /**
   * @brief algrithm 1
   * 1. 对于左图每一网格块，遍历右图寻找与i匹配对最多的块j
   * 2. 遍历i、j的3×3邻块，统计总匹配点数
   * 3. 计算阈值
   * 4. 添加内点
   *
   * @param rotationType
   */
  void verify(int rotationType) {
    const int* currRotationType = mRotationPatterns[rotationType - 1];

    for (int i = 0; i < mPixelNumPerGridLeft; i++) {
      if (cv::sum(mMotionStatistics.row(i))[0] == 0) {
        mCellPairs[i] = -1;
        continue;
      }

      int maxNums = 0;
      for (int j = 0; j < mPixelNumPerGridRight; j++) {
        int* val = mMotionStatistics.ptr<int>(i);  //第i行第一个元素的指针
        if (val[j] > maxNums) {
          mCellPairs[i] = j;
          maxNums = val[j];
        }
      }

      int rgIdx = mCellPairs[i];  //网格在右图中的索引
      const int* lNB9 = mGridNeighborLeft.ptr<int>(i);  //第i行第一个元素的指针
      const int* rNB9 =
          mGridNeighborRight.ptr<int>(rgIdx);  //第idx_grid_rt行第一个元素的指针

      int score = 0;  // S_ij = sum(|X_ik_jk|), k=1~9
      double threshold =
          0;  // = THRES_FACTOR * sqrt(average kps in single cell)
      int numsPair = 0;

      for (size_t k = 0; k < 9; k++) {
        int lgNBIdx = lNB9[k];
        int rgNBIdx = rNB9[currRotationType[k] - 1];
        if (lgNBIdx == -1 || rgNBIdx == -1) continue;

        // update
        score += mMotionStatistics.at<int>(lgNBIdx, rgNBIdx);
        threshold += mNumKpsPerCellLeft[lgNBIdx];
        numsPair++;
      }

      //? may should be threshold = THRES_FACTOR * sqrt(numsPair / 9)
      threshold = THRES_FACTOR * sqrt(threshold / numsPair);
      // threshold = THRES_FACTOR * sqrt(numsPair / 9);
      if (score < threshold) mCellPairs[i] = -2;  // outliers
    }
  }

  /**
   * @brief 实际执行函数
   *
   * @param rotationType 旋转模板编号
   * @return int
   */
  int run(int rotationType) {
    mvInlierMask.assign(mNumMatches, false);

    mMotionStatistics =
        cv::Mat::zeros(mPixelNumPerGridLeft, mPixelNumPerGridRight, CV_32SC1);
    mvMatchPairs.assign(mNumMatches, pair<int, int>(0, 0));

    for (int gridType = 1; gridType <= 4; gridType++) {
      mMotionStatistics.setTo(0);
      mCellPairs.assign(mPixelNumPerGridLeft, -1);
      mNumKpsPerCellLeft.assign(mPixelNumPerGridLeft, 0);

      assignMatchPairs(gridType);
      // cout << "end assign" << endl;
      verify(rotationType);
      // cout << "end verify" << endl;

      // mark all the inliers in the Mask
      for (size_t i = 0; i < mNumMatches; i++) {
        if (mvMatchPairs[i].first >= 0)
          if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
            mvInlierMask[i] = true;
      }
    }

    int Inliers = cv::sum(mvInlierMask)[0];
    return Inliers;
  }

 private:
  vector<cv::Point2f> mvp1, mvp2;          //归一化后的关键点
  vector<pair<int, int>> mvMatches;        //转换后的原始匹配对
  size_t mNumMatches;                      //原始匹配对数
  cv::Size mGridSizeLeft, mGridSizeRight;  //网格尺寸: 20*20
  int mPixelNumPerGridLeft, mPixelNumPerGridRight;  //每个网格中的像素点数
  vector<int> mNumKpsPerCellLeft;  //左图每个网格中的特征点数
  vector<bool> mvInlierMask;       //标记内点

  // Mat(left grid idx, right gird idx) = number of matches
  cv::Mat mMotionStatistics;

  // Every Matches has a cell-pair
  // pair<grid_idx_left, grid_idx_right>
  vector<pair<int, int>> mvMatchPairs;

  // mCellPairs[grid_idx_left] = grid_idx_right
  vector<int> mCellPairs;

  // 400*9
  cv::Mat mGridNeighborLeft, mGridNeighborRight;

  const int mRotationPatterns[8][9] = {1, 2, 3, 4, 5, 6, 7, 8, 9,

                                       4, 1, 2, 7, 5, 3, 8, 9, 6,

                                       7, 4, 1, 8, 5, 2, 9, 6, 3,

                                       8, 7, 4, 9, 5, 1, 6, 3, 2,

                                       9, 8, 7, 6, 5, 4, 3, 2, 1,

                                       6, 9, 8, 3, 5, 7, 2, 1, 4,

                                       3, 6, 9, 2, 5, 8, 1, 4, 7,

                                       2, 3, 6, 1, 5, 9, 4, 7, 8};

  const double mScaleRatios[5] = {1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0),
                                  2.0};
};

#endif