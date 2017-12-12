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

#pragma once

#include <algorithm>
#include <math.h>

#include "util.h"

// Implementation of the camera model used in the ETH3D benchmark. Supports
// conversions between the following coordinates (partially):
// 
// * Local camera coordinates (short: Point): 3D point coordinates having the
//   origin at the camera's projection center. X goes to the right, Y goes down,
//   and Z goes forward in the direction of the optical axis.
// * Normalized image coordinates (short: Normalized). This is what results if a
//   3D point in local camera coordinates is projected to the plane having z = 1
//   by intersecting a line going through the point and the origin with this
//   plane.
// * Distorted normalized image coordinates (short: NormalizedDistorted).
//   This is what results from taking the normalized image coordinates,
//   applying the image distortion, but before mapping the coordinates to
//   pixels.
// * Pixel coordinates (short: Pixel). In pixel coordinates, (0, 0) is the
//   center of the top-left pixel. The center of the bottom-right pixel is
//   (width - 1, height - 1).
// * Texture coordinates (short: Texture). In this system, (0, 0) is the
//   top-left corner of the top-left pixel, while (1, 1) is the bottom-right
//   corner of the bottom-right pixel.
// 
// The projection functions follow the naming scheme <source>To<target> using
// the short name of the coordinate systems, for example: NormalizedToPixel.
// For Unprojection (PixelToPoint and PixelToNormalized),
// InitializeUnprojectionLookup() must be called first.
class ThinPrismFisheyeCamera {
 public:
  // Constructor which takes all camera parameters separately.
  ThinPrismFisheyeCamera(int width, int height, float fx, float fy,
                         float cx, float cy, float k1, float k2,
                         float p1, float p2, float k3, float k4,
                         float sx1, float sy1);
  
  // Constructor which takes the intrinsic parameters as a pointer. It must
  // point to an array of 12 elements which are ordered in the same way
  // as in the constructor above.
  ThinPrismFisheyeCamera(int width, int height, const float* parameters);
  
  // No copy constructor implemented.
  ThinPrismFisheyeCamera(
      const ThinPrismFisheyeCamera& other) = delete;
  
  // Destructor. Frees the unprojection lookup, if it has been created.
  ~ThinPrismFisheyeCamera();
  
  // Returns the number of intrinsic parameters (excluding the image width and
  // height).
  static constexpr int ParameterCount() {
    // 4 for the pinhole model + 8 for the distortion.
    return 4 + 8;
  }
  
  // Creates the unprojection lookup. Must be called before using unprojection
  // functions.
  void InitializeUnprojectionLookup();
  
  // Returns a copy of the camera for which the image size is scaled by the
  // given factor.
  ThinPrismFisheyeCamera* ScaledBy(float factor) const;
  
  // Returns a copy of the camera for which cx and cy are offset by the given
  // values.
  ThinPrismFisheyeCamera* ShiftedBy(float cx_offset, float cy_offset) const;
  
  // Projects a point to pixel coordinates. Returns true if the point projects
  // within the image, false otherwise. The pixel coordinate bounds used for
  // this check are (-0.5, -0.5) to (width - 0.5, height - 0.5).
  inline bool
  PointToPixel(const Vec3f& point, Vec2f* result) const {
    if (point.z <= 0) {
      return false;
    }
    const float point_z_inv = 1 / point.z;
    const Vec2f normalized =
        Vec2f(point.x * point_z_inv,
              point.y * point_z_inv);
    *result = NormalizedToPixel(normalized);
    return (result->x >= -0.5f &&
            result->y >= -0.5f &&
            result->x < width() - 0.5f &&
            result->y < height() - 0.5f);
  }
  
  // Unprojects a pixel to a point with the given Z coordinate.
  inline Vec3f
  PixelToPoint(int x, int y, float point_z) const {
    const Vec2f nxy = PixelToNormalized(x, y);
    return Vec3f(point_z * nxy.x,
                 point_z * nxy.y,
                 point_z);
  }
  
  // Unprojects a pixel to a point with the given Z coordinate.
  inline Vec3f
  PixelToPoint(const Vec2f& pixel, float point_z) const {
    const Vec2f nxy = PixelToNormalized(pixel);
    return Vec3f(point_z * nxy.x,
                 point_z * nxy.y,
                 point_z);
  }
  
  inline Vec2f
  NormalizedToTexture(const Vec2f normalized_point) const {
    const Vec2f distorted_point =
        NormalizedToNormalizedDistorted(normalized_point);
    return Vec2f(nfx() * distorted_point.x + ncx(),
                 nfy() * distorted_point.y + ncy());
  }
  
  inline Vec2f
  NormalizedToPixel(const Vec2f normalized_point) const {
    const Vec2f distorted_point =
        NormalizedToNormalizedDistorted(normalized_point);
    return Vec2f(fx() * distorted_point.x + cx(),
                 fy() * distorted_point.y + cy());
  }
  
  inline Vec2f
  NormalizedToNormalizedDistorted(const Vec2f normalized_point) const {
    const float r = sqrtf(normalized_point.x * normalized_point.x +
                          normalized_point.y * normalized_point.y);
    float x, y;
    if (r > radius_cutoff_) {
      // Return something outside of the image.
      return Vec2f(9999, 9999);
    }
    if (r > kEpsilon) {
      const float theta_by_r = atan2(r, 1.f) / r;
      x = theta_by_r * normalized_point.x;
      y = theta_by_r * normalized_point.y;
    } else {
      x = normalized_point.x;
      y = normalized_point.y;
    }
    
    return NormalizedToNormalizedDistortedWithoutFisheye(Vec2f(x, y));
  }

  inline Vec2f
  PixelToNormalized(const int x, const int y) const {
    return undistortion_lookup_[y * width_ + x];
  }

  inline Vec2f
  PixelToNormalized(const Vec2f pixel_position) const {
    // Manual implementation of bilinearly filtering the lookup.
    Vec2f clamped_pixel = Vec2f(
        std::max(0.f, std::min(width() - 1.001f, pixel_position.x)),
        std::max(0.f, std::min(height() - 1.001f, pixel_position.y)));
    int int_pos_x = clamped_pixel.x;
    int int_pos_y = clamped_pixel.y;
    Vec2f factor =
        Vec2f(clamped_pixel.x - int_pos_x, clamped_pixel.y - int_pos_y);
    Vec2f top_left = undistortion_lookup_[int_pos_y * width_ + int_pos_x];
    Vec2f top_right =
        undistortion_lookup_[int_pos_y * width_ + (int_pos_x + 1)];
    Vec2f bottom_left =
        undistortion_lookup_[(int_pos_y + 1) * width_ + int_pos_x];
    Vec2f bottom_right =
        undistortion_lookup_[(int_pos_y + 1) * width_ + (int_pos_x + 1)];
    return Vec2f(
        (1 - factor.y) *
            ((1 - factor.x) * top_left.x + factor.x * top_right.x) +
            factor.y *
            ((1 - factor.x) * bottom_left.x + factor.x * bottom_right.x),
        (1 - factor.y) *
            ((1 - factor.x) * top_left.y + factor.x * top_right.y) +
            factor.y *
            ((1 - factor.x) * bottom_left.y + factor.x * bottom_right.y));
  }

  // This iterative undistortion function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the PixelToNormalized() methods. This function is only
  // used for calculating this undistortion texture once.
  inline Vec2f
  NormalizedDistortedToNormalized(const Vec2f distorted_point) const {
    const size_t kNumUndistortionIterations = 100;
    
    // Gauss-Newton.
    float uu = distorted_point.x;
    float vv = distorted_point.y;
    const float kUndistortionEpsilon = 1e-10f;
    for (size_t i = 0; i < kNumUndistortionIterations; ++i) {
      Vec2f distorted =
          NormalizedToNormalizedDistortedWithoutFisheye(Vec2f(uu, vv));
      // (Non-squared) residuals.
      float dx = distorted.x - distorted_point.x;
      float dy = distorted.y - distorted_point.y;
      
      // Accumulate H and b.
      Vec4f ddxy_dxy =
          NormalizedToNormalizedDistortedWithoutFisheyeJacobianWrtNormalized(
              Vec2f(uu, vv));
      float H_0_0 = ddxy_dxy.x * ddxy_dxy.x + ddxy_dxy.z * ddxy_dxy.z;
      float H_1_0_and_0_1 = ddxy_dxy.x * ddxy_dxy.y + ddxy_dxy.z * ddxy_dxy.w;
      float H_1_1 = ddxy_dxy.y * ddxy_dxy.y + ddxy_dxy.w * ddxy_dxy.w;
      float b_0 = dx * ddxy_dxy.x + dy * ddxy_dxy.z;
      float b_1 = dx * ddxy_dxy.y + dy * ddxy_dxy.w;
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      uu -= x_0;
      vv -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        break;
      }
    }
    
    const float theta = sqrtf(uu * uu + vv * vv);
    const float theta_cos_theta = theta * cosf(theta);
    if (theta_cos_theta > kEpsilon) {
      const float scale = sinf(theta) / theta_cos_theta;
      uu *= scale;
      vv *= scale;
    }
    
    return Vec2f(uu, vv);
  }
  
  // Returns the derivatives of the texture coordinates with
  // respect to the 3D change of the input point.
  inline
  void PointToTextureJacobian(
      const Vec3f& point, Vec3f* deriv_x, Vec3f* deriv_y) const {
    const Vec2f normalized_point =
        Vec2f(point.x / point.z, point.y / point.z);
    const Vec4f distortion_deriv =
        NormalizedToNormalizedDistortedJacobianWrtNormalized(normalized_point);
    const Vec4f projection_deriv =
        Vec4f(nfx() * distortion_deriv.x,
              nfx() * distortion_deriv.y,
              nfy() * distortion_deriv.z,
              nfy() * distortion_deriv.w);
    *deriv_x = Vec3f(
        projection_deriv.x / point.z, projection_deriv.y / point.z,
        -1.0f * (projection_deriv.x * point.x + projection_deriv.y * point.y) /
            (point.z * point.z));
    *deriv_y = Vec3f(
        projection_deriv.z / point.z, projection_deriv.w / point.z,
        -1.0f * (projection_deriv.z * point.x + projection_deriv.w * point.y) /
            (point.z * point.z));
  }
  
  // Returns the derivatives of the pixel coordinates with
  // respect to the 3D change of the input point.
  inline
  void PointToPixelJacobianWrtPoint(
      const Vec3f& point, Vec3f* deriv_x, Vec3f* deriv_y) const {
    const Vec2f normalized_point =
        Vec2f(point.x / point.z, point.y / point.z);
    const Vec4f distortion_deriv =
        NormalizedToNormalizedDistortedJacobianWrtNormalized(normalized_point);
    const Vec4f projection_deriv =
        Vec4f(fx() * distortion_deriv.x,
              fx() * distortion_deriv.y,
              fy() * distortion_deriv.z,
              fy() * distortion_deriv.w);
    *deriv_x = Vec3f(
        projection_deriv.x / point.z, projection_deriv.y / point.z,
        -1.0f * (projection_deriv.x * point.x + projection_deriv.y * point.y) /
            (point.z * point.z));
    *deriv_y = Vec3f(
        projection_deriv.z / point.z, projection_deriv.w / point.z,
        -1.0f * (projection_deriv.z * point.x + projection_deriv.w * point.y) /
            (point.z * point.z));
  }
  
  // Returns the derivatives of the pixel coordinates with respect to the
  // intrinsics. For x and y, 12 values each are returned for fx, fy, cx, cy,
  // k1, k2, p1, p2, k3, k4, sx1, sy1.
  inline
  void PointToPixelJacobianWrtIntrinsics(
      const Vec3f& point, float* deriv_x, float* deriv_y) const {
    const Vec2f normalized_point =
        Vec2f(point.x / point.z, point.y / point.z);
    
    const Vec2f distorted_point =
        NormalizedToNormalizedDistorted(normalized_point);
    deriv_x[0] = distorted_point.x;
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y;
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    
    const float nx = normalized_point.x;
    const float ny = normalized_point.y;
    const float nx2 = normalized_point.x * normalized_point.x;
    const float ny2 = normalized_point.y * normalized_point.y;
    const float two_nx_ny = 2.f * nx * ny;
    const float fx_nx = fx() * normalized_point.x;
    const float fy_ny = fy() * normalized_point.y;
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > radius_cutoff_) {
      for (int i = 0; i < 12; ++ i) {
        deriv_x[i] = 0;
        deriv_y[i] = 0;
      }
      return;
    }
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float atan_r_2 = atan_r * atan_r;
      const float atan_r_3_by_r = (atan_r_2 * atan_r) / r;
      const float two_nx_ny_atan_r_2_by_r2 = (two_nx_ny * atan_r_2) / r2;
      const float atan_r_2_by_r2 = atan_r_2 / r2;
      
      deriv_x[4] = fx_nx * atan_r_3_by_r;
      deriv_x[5] = deriv_x[4] * atan_r_2;
      deriv_x[6] = fx() * two_nx_ny_atan_r_2_by_r2;
      deriv_x[7] = fx() * atan_r_2_by_r2 * (3 * nx2 + ny2);
      deriv_x[8] = deriv_x[5] * atan_r_2;
      deriv_x[9] = deriv_x[8] * atan_r_2;
      deriv_x[10] = fx() * atan_r_2;
      deriv_x[11] = 0;
      
      deriv_y[4] = fy_ny * atan_r_3_by_r;
      deriv_y[5] = deriv_y[4] * atan_r_2;
      deriv_y[6] = fy() * atan_r_2_by_r2 * (nx2 + 3 * ny2);
      deriv_y[7] = fy() * two_nx_ny_atan_r_2_by_r2;
      deriv_y[8] = deriv_y[5] * atan_r_2;
      deriv_y[9] = deriv_y[8] * atan_r_2;
      deriv_y[10] = 0;
      deriv_y[11] = fy() * atan_r_2;
    } else {
      // The non-fisheye variant is used in this case.
      deriv_x[4] = fx_nx * r2;
      deriv_x[5] = deriv_x[4] * r2;
      deriv_x[6] = fx() * two_nx_ny;
      deriv_x[7] = fx() * (r2 + 2.f * nx2);
      deriv_x[8] = deriv_x[5] * r2;
      deriv_x[9] = deriv_x[8] * r2;
      deriv_x[10] = fx() * r2;
      deriv_x[11] = 0;
      
      deriv_y[4] = fy_ny * r2;
      deriv_y[5] = deriv_y[4] * r2;
      deriv_y[6] = fy() * (r2 + 2.f * ny2);
      deriv_y[7] = fy() * two_nx_ny;
      deriv_y[8] = deriv_y[5] * r2;
      deriv_y[9] = deriv_y[8] * r2;
      deriv_y[10] = 0;
      deriv_y[11] = fy() * r2;
    }
  }
  
  // Jacobian of the distortion wrt. the normalized coordinates.
  inline Vec4f
  NormalizedToNormalizedDistortedJacobianWrtNormalized(
      const Vec2f& normalized_point) const {
    const float nx = normalized_point.x;
    const float ny = normalized_point.y;
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > radius_cutoff_) {
      return Vec4f(0, 0, 0, 0);
    }
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float r3 = r2 * r;
      
      const float term1 = r2 * (r2 + 1);
      const float term2 = atan_r / r3;
      
      // Derivatives of fisheye x / y coordinates by nx / ny:
      const float dnxf_dnx = ny2 * term2 + nx2 / term1;
      const float dnxf_dny = nx_ny / term1 - nx_ny * term2;
      const float dnyf_dnx = dnxf_dny;
      const float dnyf_dny = nx2 * term2 + ny2 / term1;
      
      // Compute fisheye x / y.
      const float theta_by_r = atan2(r, 1.f) / r;
      const float x = theta_by_r * nx;
      const float y = theta_by_r * ny;
      
      // Derivatives of distorted coordinates by fisheye x / y:
      // (same computation as in non-fisheye polynomial-tangential)

      const float x_y = x * y;
      const float x2 = x * x;
      const float y2 = y * y;
      
      const float rf2 = x2 + y2;
      const float rf4 = rf2 * rf2;
      const float rf6 = rf4 * rf2;
      const float rf8 = rf6 * rf2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1f = 2*p1*x + 2*p2*y + 2*k1*x_y + 6*k3*x_y*rf4 + 8*k4*x_y*rf6 + 4*k2*x_y*rf2;
      const float ddx_dnxf = 2*k1*x2 + 4*k2*x2*rf2 + 6*k3*x2*rf4 + 8*k4*x2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 6*p2*x + 2*p1*y + 2*sx1*x + k1*rf2 + 1;
      const float ddx_dnyf = 2*sx1*y + term1f;
      const float ddy_dnxf = 2*sy1*x + term1f;
      const float ddy_dnyf = 2*k1*y2 + 4*k2*y2*rf2 + 6*k3*y2*rf4 + 8*k4*y2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 2*p2*x + 6*p1*y + 2*sy1*y + k1*rf2 + 1;
      return Vec4f(ddx_dnxf * dnxf_dnx + ddx_dnyf * dnyf_dnx,
                   ddy_dnxf * dnxf_dnx + ddy_dnyf * dnyf_dnx,
                   ddx_dnxf * dnxf_dny + ddx_dnyf * dnyf_dny,
                   ddy_dnxf * dnxf_dny + ddy_dnyf * dnyf_dny);
    } else {
      // Non-fisheye variant is used in this case.
      const float r4 = r2 * r2;
      const float r6 = r4 * r2;
      const float r8 = r6 * r2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
      const float ddx_dnx = 2*k1*nx2 + 4*k2*nx2*r2 + 6*k3*nx2*r4 + 8*k4*nx2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
      const float ddx_dny = 2*sx1*ny + term1;
      const float ddy_dnx = 2*sy1*nx + term1;
      const float ddy_dny = 2*k1*ny2 + 4*k2*ny2*r2 + 6*k3*ny2*r4 + 8*k4*ny2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
      return Vec4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
    }
  }
  
  // Copies the intrinsic camera parameters into an array, in the same order as
  // taken by the constructor.
  inline void
  GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_[0];
    parameters[5] = distortion_parameters_[1];
    parameters[6] = distortion_parameters_[2];
    parameters[7] = distortion_parameters_[3];
    parameters[8] = distortion_parameters_[4];
    parameters[9] = distortion_parameters_[5];
    parameters[10] = distortion_parameters_[6];
    parameters[11] = distortion_parameters_[7];
  }
  
  // Returns the image width in pixels.
  inline int width() const { return width_; }
  
  // Returns the image height in pixels.
  inline int height() const { return height_; }
  
  // Returns the fx coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (0, 0)).
  inline float fx() const { return k_.x; }
  
  // Returns the fy coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (1, 1)).
  inline float fy() const { return k_.y; }
  
  // Returns the cx coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (0, 2)).
  inline float cx() const { return k_.z; }
  
  // Returns the cy coefficient of the intrinsics matrix for image coordinates
  // (matrix entry (1, 2)).
  inline float cy() const { return k_.w; }
  
  // Returns the fx coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (0, 0)).
  inline float nfx() const {
    return normalized_k_.x;
  }
  
  // Returns the fy coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (1, 1)).
  inline float nfy() const {
    return normalized_k_.y;
  }
  
  // Returns the cx coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (0, 2)).
  inline float ncx() const {
    return normalized_k_.z;
  }
  
  // Returns the cy coefficient of the normalized intrinsics matrix for image
  // coordinates (matrix entry (1, 2)).
  inline float ncy() const {
    return normalized_k_.w;
  }
  
  // Returns the fx coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (0, 0)).
  inline float fx_inv() const { return k_inv_.x; }
  
  // Returns the fy coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (1, 1)).
  inline float fy_inv() const { return k_inv_.y; }
  
  // Returns the cx coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (0, 2)).
  inline float cx_inv() const { return k_inv_.z; }
  
  // Returns the cy coefficient of the inverse intrinsics matrix for image
  // coordinates (matrix entry (1, 2)).
  inline float cy_inv() const { return k_inv_.w; }
  
  // Returns the distortion parameters.
  inline const float* distortion_parameters() const {
    return distortion_parameters_;
  }
  
  // Returns the radius cutoff value.
  inline float radius_cutoff() const {
    return radius_cutoff_;
  }

 private:
  void InitCutoff();
  
  inline Vec2f
  NormalizedToNormalizedDistortedWithoutFisheye(
      const Vec2f normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float x2 = normalized_point.x * normalized_point.x;
    const float xy = normalized_point.x * normalized_point.y;
    const float y2 = normalized_point.y * normalized_point.y;
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    const float radial =
        k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    const float dx = 2.f * p1 * xy + p2 * (r2 + 2.f * x2) + sx1 * r2;
    const float dy = 2.f * p2 * xy + p1 * (r2 + 2.f * y2) + sy1 * r2;
    return Vec2f(
        normalized_point.x + radial * normalized_point.x + dx,
        normalized_point.y + radial * normalized_point.y + dy);
  }
  
  inline Vec4f
  NormalizedToNormalizedDistortedWithoutFisheyeJacobianWrtNormalized(
      const Vec2f& normalized_point) const {
    const float nx = normalized_point.x;
    const float ny = normalized_point.y;
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;

    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    // NOTE: Could factor out more terms here which might improve performance.
    const float term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
    const float ddx_dnx = 2*k1*nx2 + 4*k2*nx2*r2 + 6*k3*nx2*r4 + 8*k4*nx2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
    const float ddx_dny = 2*sx1*ny + term1;
    const float ddy_dnx = 2*sy1*nx + term1;
    const float ddy_dny = 2*k1*ny2 + 4*k2*ny2*r2 + 6*k3*ny2*r4 + 8*k4*ny2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
    return Vec4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  // Intrinsics.
  // Values are stored in the order: fx, fy, cx, cy.
  Vec4f normalized_k_;
  Vec4f k_;
  Vec4f k_inv_;
  
  // Image size.
  int width_;
  int height_;
  
  // The distortion parameters k1, k2, p1, p2, k3, k4, sx1, sy1.
  float distortion_parameters_[8];
  
  Vec2f* undistortion_lookup_;
  
  float radius_cutoff_;
  
  static constexpr float kEpsilon = 1e-6f;
};
