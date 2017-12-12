// Copyright 2017 Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "camera.h"
#include "util.h"


// Unit test helpers.
template<typename T>
bool ExpectEqual(T a, const char* a_name,
                 T b, const char* b_name,
                 const char* file, int line) {
  if (a != b) {
    std::cerr << "EXPECT_EQUAL failed: " << a_name << " (" << a << ")"
              << " != " << b_name << " (" << b << ") at " << file
              << " (line " << line << ")" << std::endl;
    return false;
  }
  return true;
}
#define EXPECT_EQUAL(a, b) \
    ExpectEqual(a, #a, b, #b, __FILE__, __LINE__)

template<typename T>
bool ExpectNear(T a, const char* a_name,
                T b, const char* b_name,
                T distance, const char* distance_name,
                const char* file, int line) {
  if (fabs(a - b) > distance) {
    std::cerr << "EXPECT_NEAR failed: fabs(" << a_name << " (" << a << ")"
              << " - " << b_name << " (" << b << ")) > " << distance_name
              << " (" << distance << ") at " << file << " (line " << line
              << ")" << std::endl;
    return false;
  }
  return true;
}
#define EXPECT_NEAR(a, b, distance) \
    ExpectNear(a, #a, b, #b, distance, #distance, __FILE__, __LINE__)


template<class Camera>
void TestParameterStorage() {
  constexpr int kNumParameters = Camera::ParameterCount();

  float parameters[kNumParameters];
  for (int i = 0; i < kNumParameters; ++ i) {
    parameters[i] = 10 * i;
  }
  Camera camera(100, 100, parameters);

  float results[kNumParameters];
  camera.GetParameters(results);
  for (int i = 0; i < kNumParameters; ++ i) {
    EXPECT_EQUAL(parameters[i], results[i]);
  }
}

template <typename Camera>
void UndistortAndDistortImageCornersTest(const Camera& test_camera) {
  // Test the corners of the image.
  std::vector<Vec2f> test_points;
  test_points.push_back(Vec2f(0, 0));
  test_points.push_back(Vec2f(test_camera.width() - 1, 0));
  test_points.push_back(Vec2f(0, test_camera.height() - 1));
  test_points.push_back(Vec2f(test_camera.width() - 1,
                              test_camera.height() - 1));

  for (size_t i = 0; i < test_points.size(); ++i) {
    Vec2f nxy = Vec2f(
        test_camera.fx_inv() * test_points[i].x + test_camera.cx_inv(),
        test_camera.fy_inv() * test_points[i].y + test_camera.cy_inv());
    Vec2f result =
        test_camera.NormalizedToNormalizedDistorted(
            test_camera.NormalizedDistortedToNormalized(nxy));

    if (!EXPECT_NEAR(nxy.x, result.x, 1e-5f)) {
      std::cerr << "Test failed for " << test_points[i].x << ", "
                << test_points[i].y << " (undistorted: "
                << test_camera.NormalizedDistortedToNormalized(nxy).x << ", "
                << test_camera.NormalizedToNormalizedDistorted(nxy).y
                << ")" << std::endl;
    }
    if (!EXPECT_NEAR(nxy.y, result.y, 1e-5f)) {
      std::cerr << "Test failed for " << test_points[i].x << ", "
                << test_points[i].y << " (undistorted: "
                << test_camera.NormalizedDistortedToNormalized(nxy).x << ", "
                << test_camera.NormalizedToNormalizedDistorted(nxy).y
                << ")" << std::endl;
    }
  }
}

template <typename Camera>
void NumericalPointToPixelJacobianWrtPoint(
    const Camera& camera, const Vec3f& at, Vec3f* out_x, Vec3f* out_y) {
  const float kStep = 0.001f;
  const float kTwoSteps = 2 * kStep;

  Vec3f at_plus_x = Vec3f(at.x + kStep, at.y, at.z);
  Vec2f proj_plus_x = camera.NormalizedToPixel(
      Vec2f(at_plus_x.x / at_plus_x.z, at_plus_x.y / at_plus_x.z));
  Vec3f at_minus_x = Vec3f(at.x - kStep, at.y, at.z);
  Vec2f proj_minus_x = camera.NormalizedToPixel(
      Vec2f(at_minus_x.x / at_minus_x.z, at_minus_x.y / at_minus_x.z));
  Vec3f at_plus_y = Vec3f(at.x, at.y + kStep, at.z);
  Vec2f proj_plus_y = camera.NormalizedToPixel(
      Vec2f(at_plus_y.x / at_plus_y.z, at_plus_y.y / at_plus_y.z));
  Vec3f at_minus_y = Vec3f(at.x, at.y - kStep, at.z);
  Vec2f proj_minus_y = camera.NormalizedToPixel(
      Vec2f(at_minus_y.x / at_minus_y.z, at_minus_y.y / at_minus_y.z));
  Vec3f at_plus_z = Vec3f(at.x, at.y, at.z + kStep);
  Vec2f proj_plus_z = camera.NormalizedToPixel(
      Vec2f(at_plus_z.x / at_plus_z.z, at_plus_z.y / at_plus_z.z));
  Vec3f at_minus_z = Vec3f(at.x, at.y, at.z - kStep);
  Vec2f proj_minus_z = camera.NormalizedToPixel(
      Vec2f(at_minus_z.x / at_minus_z.z, at_minus_z.y / at_minus_z.z));

  *out_x = Vec3f((proj_plus_x.x - proj_minus_x.x) / kTwoSteps,
                 (proj_plus_y.x - proj_minus_y.x) / kTwoSteps,
                 (proj_plus_z.x - proj_minus_z.x) / kTwoSteps);
  *out_y = Vec3f((proj_plus_x.y - proj_minus_x.y) / kTwoSteps,
                 (proj_plus_y.y - proj_minus_y.y) / kTwoSteps,
                 (proj_plus_z.y - proj_minus_z.y) / kTwoSteps);
}

template <typename Camera>
void CheckPointToPixelJacobianWrtPoint(const Camera& camera,
                                       const Vec3f& at) {
  Vec3f result_x, result_y;
  camera.PointToPixelJacobianWrtPoint(at, &result_x, &result_y);
  Vec3f numerical_x, numerical_y;
  NumericalPointToPixelJacobianWrtPoint(camera, at, &numerical_x, &numerical_y);

  bool success = true;
  success &= EXPECT_NEAR(result_x.x, numerical_x.x, 250 * 1e-3f);
  success &= EXPECT_NEAR(result_x.y, numerical_x.y, 250 * 1e-3f);
  success &= EXPECT_NEAR(result_x.z, numerical_x.z, 250 * 1e-3f);
  success &= EXPECT_NEAR(result_y.x, numerical_y.x, 250 * 1e-3f);
  success &= EXPECT_NEAR(result_y.y, numerical_y.y, 250 * 1e-3f);
  success &= EXPECT_NEAR(result_y.z, numerical_y.z, 250 * 1e-3f);
  if (!success) {
    std::cerr << "Test failed for point " << at.x << ", " << at.y << ", "
              << at.z << std::endl;
  }
}

template <typename Camera>
void TestPointToPixelJacobianWrtPoint(const Camera& camera) {
  const Vec3f kTest3DPoints[] = {Vec3f(0.0f, 0.0f, 3.0f),
                                 Vec3f(1.0f, 3.0f, 8.0f),
                                 Vec3f(-0.1f, 0.7f, -0.8f)};
  for (size_t i = 0;
       i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]);
       ++i) {
    CheckPointToPixelJacobianWrtPoint(camera, kTest3DPoints[i]);
  }
}

template <typename Camera>
void NumericalPointToTextureJacobian(
    const Camera& camera, const Vec3f& at, Vec3f* out_x, Vec3f* out_y) {
  const float kStep = 0.001f;
  const float kTwoSteps = 2 * kStep;

  Vec3f at_plus_x = Vec3f(at.x + kStep, at.y, at.z);
  Vec2f proj_plus_x = camera.NormalizedToTexture(
      Vec2f(at_plus_x.x / at_plus_x.z, at_plus_x.y / at_plus_x.z));
  Vec3f at_minus_x = Vec3f(at.x - kStep, at.y, at.z);
  Vec2f proj_minus_x = camera.NormalizedToTexture(
      Vec2f(at_minus_x.x / at_minus_x.z, at_minus_x.y / at_minus_x.z));
  Vec3f at_plus_y = Vec3f(at.x, at.y + kStep, at.z);
  Vec2f proj_plus_y = camera.NormalizedToTexture(
      Vec2f(at_plus_y.x / at_plus_y.z, at_plus_y.y / at_plus_y.z));
  Vec3f at_minus_y = Vec3f(at.x, at.y - kStep, at.z);
  Vec2f proj_minus_y = camera.NormalizedToTexture(
      Vec2f(at_minus_y.x / at_minus_y.z, at_minus_y.y / at_minus_y.z));
  Vec3f at_plus_z = Vec3f(at.x, at.y, at.z + kStep);
  Vec2f proj_plus_z = camera.NormalizedToTexture(
      Vec2f(at_plus_z.x / at_plus_z.z, at_plus_z.y / at_plus_z.z));
  Vec3f at_minus_z = Vec3f(at.x, at.y, at.z - kStep);
  Vec2f proj_minus_z = camera.NormalizedToTexture(
      Vec2f(at_minus_z.x / at_minus_z.z, at_minus_z.y / at_minus_z.z));

  *out_x = Vec3f((proj_plus_x.x - proj_minus_x.x) / kTwoSteps,
                 (proj_plus_y.x - proj_minus_y.x) / kTwoSteps,
                 (proj_plus_z.x - proj_minus_z.x) / kTwoSteps);
  *out_y = Vec3f((proj_plus_x.y - proj_minus_x.y) / kTwoSteps,
                 (proj_plus_y.y - proj_minus_y.y) / kTwoSteps,
                 (proj_plus_z.y - proj_minus_z.y) / kTwoSteps);
}

template <typename Camera>
void CheckPointToTextureJacobian(
    const Camera& camera, const Vec3f& at) {
  Vec3f result_x, result_y;
  camera.PointToTextureJacobian(at, &result_x, &result_y);
  Vec3f numerical_x, numerical_y;
  NumericalPointToTextureJacobian(camera, at, &numerical_x, &numerical_y);

  bool success = true;
  success &= EXPECT_NEAR(result_x.x, numerical_x.x, 1e-3f);
  success &= EXPECT_NEAR(result_x.y, numerical_x.y, 1e-3f);
  success &= EXPECT_NEAR(result_x.z, numerical_x.z, 1e-3f);
  success &= EXPECT_NEAR(result_y.x, numerical_y.x, 1e-3f);
  success &= EXPECT_NEAR(result_y.y, numerical_y.y, 1e-3f);
  success &= EXPECT_NEAR(result_y.z, numerical_y.z, 1e-3f);
  if (!success) {
    std::cerr << "Test failed for point " << at.x << ", " << at.y << ", "
              << at.z << std::endl;
  }
}

template<class Camera>
void TestPointToTextureJacobian(
    const Camera& camera) {
  const Vec3f kTest3DPoints[] = {Vec3f(0.0f, 0.0f, 3.0f),
                                 Vec3f(1.0f, 3.0f, 8.0f),
                                 Vec3f(-0.1f, 0.7f, -0.8f)};
  for (size_t i = 0;
       i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]);
       ++i) {
    CheckPointToTextureJacobian(camera, kTest3DPoints[i]);
  }
}

template<class Camera>
void CreateDeltaCamera(
    const Camera& base_camera,
    const float* delta,
    std::shared_ptr<Camera>* result) {
  constexpr int kParameterCount = Camera::ParameterCount();
  float parameters[kParameterCount];
  base_camera.GetParameters(parameters);
  for (int i = 0; i < kParameterCount; ++ i) {
    parameters[i] += delta[i];
  }
  result->reset(new Camera(base_camera.width(),
                           base_camera.height(),
                           parameters));
}

template<class Camera>
void NumericalPointToPixelJacobianWrtIntrinsics(
    const Camera& base_camera,
    const float nx, const float ny,
    const float* delta,
    float* result_x, float* result_y) {
  constexpr int kNumParameters = Camera::ParameterCount();
  const float kStep = 0.025f;
  const float kTwoSteps = 2 * kStep;

  std::shared_ptr<Camera> plus, minus;
  float plus_delta[kNumParameters], minus_delta[kNumParameters];
  for (int i = 0; i < kNumParameters; ++ i) {
    plus_delta[i] = kStep * delta[i];
    minus_delta[i] = -1.f * plus_delta[i];
  }
  CreateDeltaCamera(base_camera, plus_delta, &plus);
  CreateDeltaCamera(base_camera, minus_delta, &minus);

  Vec2f proj_plus_x = plus->NormalizedToPixel(Vec2f(nx, ny));
  Vec2f proj_minus_x = minus->NormalizedToPixel(Vec2f(nx, ny));
  *result_x = (proj_plus_x.x - proj_minus_x.x) / kTwoSteps;
  *result_y = (proj_plus_x.y - proj_minus_x.y) / kTwoSteps;
}

template<class Camera>
void TestPointToPixelJacobianWrtIntrinsics(
    const Camera& camera) {
  constexpr int kNumParameters = Camera::ParameterCount();
  const Vec3f kTest3DPoints[] = {Vec3f(0.0f, 0.0f, 3.0f),
                                 Vec3f(1.0f, 2.5f, 4.0f),
                                 Vec3f(1.0f, 3.0f, 8.0f),
                                 Vec3f(-0.1f, 0.4f, 0.8f)};
  float result_x[kNumParameters], result_y[kNumParameters];
  for (unsigned int i = 0; i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]);
      ++i) {
    const float nx = kTest3DPoints[i].x / kTest3DPoints[i].z;
    const float ny = kTest3DPoints[i].y / kTest3DPoints[i].z;

    Vec2f pxy = camera.NormalizedToPixel(Vec2f(nx, ny));
    if (pxy.x < 0 || pxy.y < 0 ||
        pxy.x >= camera.width() ||
        pxy.y >= camera.height()) {
      std::cerr << "Warning: Test point " << i
                << " does not project into the camera image." << std::endl;
      continue;
    }

    camera.PointToPixelJacobianWrtIntrinsics(
        kTest3DPoints[i], result_x, result_y);

    float numerical_x, numerical_y;
    float delta[kNumParameters];
    memset(delta, 0, kNumParameters * sizeof(delta[0]));
    for (int c = 0; c < kNumParameters; ++ c) {
      delta[c] = 1;
      NumericalPointToPixelJacobianWrtIntrinsics(
            camera, nx, ny,
            delta, &numerical_x, &numerical_y);
      delta[c] = 0;

      if (!EXPECT_NEAR(result_x[c], numerical_x, 1.5e-3f)) {
        std::cerr << "Failure for point " << i << ", component " << c
                  << std::endl;
      }
      if (!EXPECT_NEAR(result_y[c], numerical_y, 1.5e-3f)) {
        std::cerr << "Failure for point " << i << ", component " << c
                  << std::endl;
      }
    }
  }
}


// Tests the camera model.
int main(int argc, char** argv) {
  ThinPrismFisheyeCamera
      test_camera(
          640, 480, 340.926, 341.124, 302.4, 201.6,
          0.221184, 0.128597, 0.000531602, -0.000388873, 0.0623079,
          0.20419, -0.000805024, 4.07704e-05);

  // Tests whether parameters are correctly set and retrieved.
  std::cout << "Running TestParameterStorage() ..." << std::endl;
  TestParameterStorage<ThinPrismFisheyeCamera>();

  // Tests whether undistortion followed by distortion results in the identity
  // function.
  std::cout << "Running UndistortAndDistortImageCornersTest() ..."
            << std::endl;
  UndistortAndDistortImageCornersTest(test_camera);

  // Compares the analytical projection-to-image-coordinates derivative to
  // numerical derivatives.
  std::cout << "Running TestPointToPixelJacobianWrtPoint() ..."
            << std::endl;
  TestPointToPixelJacobianWrtPoint(test_camera);

  // Compares the analytical projection-to-normalized-texture-coordinates
  // derivative to numerical derivatives.
  std::cout << "Running"
            << " TestPointToTextureJacobian() ..."
            << std::endl;
  TestPointToTextureJacobian(test_camera);

  // Compares the analytical projection-to-image-coordinates derivative by
  // the intrinics to numerical derivatives.
  std::cout << "Running"
            << " TestPointToPixelJacobianWrtIntrinsics() ..."
            << std::endl;
  TestPointToPixelJacobianWrtIntrinsics(test_camera);

  std::cout << "Finished (if there were no error messages,"
            << " all tests were successful)." << std::endl;
  return EXIT_SUCCESS;
}
