#ifndef SHOT_DESCRIPTOR_H
#define SHOT_DESCRIPTOR_H

#include "mesh.h"
#include <string>

class invalid_mesh_descriptor : public std::logic_error {
public:
   explicit invalid_mesh_descriptor() : std::logic_error("Exception invalid_mesh_descriptor caught.") {}
   invalid_mesh_descriptor(const std::string& msg) : std::logic_error("Exception invalid_mesh_descriptor caught: "+msg) {}
};

/**
 * Generic pointwise local shape descriptor for meshes.
 */
template <typename T>
class mesh_descriptor
{
   protected:
      T point;

   public:
      explicit mesh_descriptor() {}
      mesh_descriptor(const T& p) : point(p) {}
};

/**
 * Generic pointwise indexed vector descriptor.
 */
template <typename T>
struct vec_descriptor : public mesh_descriptor<int>
{
protected:
   std::vector<T> d;

public:
   typedef T value_type;

   explicit vec_descriptor() : mesh_descriptor<int>() {}
   explicit vec_descriptor(size_t n) : mesh_descriptor<int>() { d.resize(n, T(0)); }

   const T& operator()(size_t k) const { return d[k]; }
   T& operator()(size_t k) { return d[k]; }

   const T& at(size_t k) const { return d.at(k); }
   T& at(size_t k) { return d.at(k); }

   size_t size() const { return d.size(); }
   bool empty() const { return d.empty(); }
   void clear() { d.clear(); }
   void resize(size_t s) { d.resize(s); }
   void resize(size_t s, T v) { d.resize(s, v); }

   bool operator==(const vec_descriptor<T>& other) const
   {
      if (this == &other) return true;
      return d == other.d;
   }

   bool operator!=(const vec_descriptor<T>& other) const
   {
      if (this == &other) return false;
      return !(*this == other);
   }

   vec_descriptor<T> operator+(const vec_descriptor<T>& other) const
   {
      const size_t s = size();
      if (s != other.size())
         throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

      vec_descriptor<T> sum;
      for (size_t k=0; k<s; ++k) sum[k] = this->d[k] + other.d[k];

      return sum;
   }

   vec_descriptor<T> operator-(const vec_descriptor<T>& other) const
   {
      const size_t s = size();
      if (s != other.size())
         throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

      vec_descriptor<T> sum;
      for (size_t k=0; k<s; ++k) sum[k] = this->d[k] - other.d[k];

      return sum;
   }

   vec_descriptor<T>& operator+=(const vec_descriptor<T>& other)
                                {
      const size_t s = size();
      if (s != other.size())
         throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

      for (size_t k=0; k<s; ++k) this->d[k] += other.d[k];

      return *this;
   }

   vec_descriptor<T>& operator-=(const vec_descriptor<T>& other)
                                {
      const size_t s = size();
      if (s != other.size())
         throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

      for (size_t k=0; k<s; ++k) this->d[k] -= other.d[k];

      return *this;
   }

   template <typename S>
   vec_descriptor<T>& operator/=(const S& val)
                                      {
      const size_t s = size();
      for (size_t k=0; k<s; ++k) this->d[k] /= T(val);
      return *this;
   }

   friend std::ostream& operator<<(std::ostream& s, const vec_descriptor& d)
   {
      s << d(0);
      for (size_t k=1; k < d.size(); ++k) s << " " << d(k);
      return s;
   }
};

namespace unibo {

class shot : public vec_descriptor<float>
{
   typedef vec_descriptor<float> super;

   public:
      typedef super::value_type value_type;

      explicit shot() : super(), radius(0.0f) {}
      explicit shot(size_t n) : super(n), radius(0.0f) {}

      float radius;
};

struct SHOTParams
{
   float radius;			///< radius of the sphere that defines the local neighborhood
   float localRFradius;	///< radius of the support to be used for the estimation of the local RF
   int bins;				///< quantization bins for the cosine
   bool doubleVolumes;
   bool useInterpolation;
   bool useNormalization;
   int minNeighbors;

   SHOTParams()	// default values
   {
      radius = 15;
      localRFradius = radius;
      bins = 10;
      minNeighbors = 10;

      doubleVolumes = true;
      useInterpolation = true;
      useNormalization = true;
   }
};

class SHOTDescriptor
{
public:
   explicit SHOTDescriptor(const SHOTParams& params) : m_params(params)
   {
      if (m_params.doubleVolumes) m_k = 32;
      else m_k=16; //number of onion husks
      m_descLength = m_k*(m_params.bins+1);
   }

   void describe(mesh_t& data, int feat_index, shot& desc) const;

   int getDescriptorLength() const { return m_descLength; }

private:
   SHOTParams m_params;
   int m_k; //number of onion husks
   int m_descLength;

   void getSHOTLocalRF(
         mesh_t& data,
         int p,
         double radius,
         vec3d<double>& X,
         vec3d<double>& Y,
         vec3d<double>& Z) const;

   void getSHOTLocalRF(
         mesh_t& data,
         int p,
         const std::vector<int>& pts,
         const std::vector<double>& dists,
         double radius,
         vec3d<double>& X,
         vec3d<double>& Y,
         vec3d<double>& Z) const;
};

} // namespace unibo

#endif // SHOT_DESCRIPTOR_H
