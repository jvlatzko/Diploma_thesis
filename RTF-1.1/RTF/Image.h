/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Bagged.h
 * Implements a (possibly multi-dimensional) image class.
 *
 */

#ifndef H_RTF_IMAGE_H
#define H_RTF_IMAGE_H

#include <memory>
#include <cstring>

class Image
{
public:
    Image() : m_pBits(0), m_pFree(0), m_nWidth(0), m_nHeight(0), m_nBands(0), m_nStrideBytes(0), m_nElementBytes(0)
    {
    }

    Image(const Image& obj) : m_pBits(0), m_pFree(0), m_nWidth(0), m_nHeight(0), m_nBands(0), m_nStrideBytes(0), m_nElementBytes(0)
    {
        Create(obj.Width(), obj.Height(), obj.ElementBytes(), obj.Bands());
    }

    virtual ~Image()
    {
        Destroy();
    }

    void Destroy()
    {
        delete [] m_pFree;
        m_pFree = m_pBits = 0;
        m_nStrideBytes = m_nElementBytes = m_nBands = m_nWidth = m_nHeight = 0;
    }

    void Create(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned int nRowAlignBytes = 4)
    {
        if(nWidth == m_nWidth && m_nHeight == nHeight && m_nElementBytes == nElementBytes && m_nBands == nBands)
            return;

        Destroy();
        size_t nPixelBytes  = nElementBytes * nBands;
        size_t nImgRowBytes = nPixelBytes * nWidth;
        size_t nStrideBytes = (nImgRowBytes + nRowAlignBytes - 1) & ~(nRowAlignBytes - 1);
        size_t nImgBytes = nStrideBytes * nHeight + nRowAlignBytes;

        m_pFree = new unsigned char [nImgBytes];
        m_pBits = m_pFree + nRowAlignBytes - ((std::ptrdiff_t)(m_pFree) & (nRowAlignBytes - 1));
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
    }

    void Attach(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pBits = pBits;
        m_pFree = 0;
    }

    void Reference(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pBits = pBits;
        m_pFree = 0;
    }

    void AttachData(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pFree = m_pBits = pBits;
    }

    unsigned char* DetachData()
    {
        unsigned char* rv = m_pFree;
        m_pFree = 0;
        return rv;
    }

    void Clear()
    {
        std::memset(m_pBits, 0, (m_nHeight - 1) * m_nStrideBytes);
        std::memset(m_pBits + (m_nHeight - 1) * m_nStrideBytes, 0, m_nWidth * m_nBands * m_nElementBytes);
    }

    int Width() const
    {
        return m_nWidth;
    }

    int Height() const
    {
        return m_nHeight;
    }

    size_t ElementBytes() const
    {
        return m_nElementBytes;
    }
    size_t StrideBytes() const
    {
        return m_nStrideBytes;
    }
    size_t PixelBytes() const
    {
        return m_nBands * m_nElementBytes;
    }
    unsigned int Bands() const
    {
        return m_nBands;
    }
    size_t Bpp() const
    {
        return PixelBytes() * 8;
    }

    unsigned char* BytePtr()
    {
        return m_pBits;
    }
    const unsigned char* BytePtr() const
    {
        return m_pBits;
    }

protected:
    unsigned char* m_pFree;
    unsigned char* m_pBits;
    int m_nWidth, m_nHeight;
    unsigned int m_nBands;
    size_t m_nElementBytes, m_nStrideBytes;
};

template <typename T, unsigned int nBands = 1>
class ImageT : public Image
{
public:
    ImageT()
    {
    }

    ImageT(int nWidth, int nHeight)
    {
        Create(nWidth, nHeight);
    }

    ImageT(int nWidth, int nHeight, unsigned int nRowAlignBytes)
    {
        Create(nWidth, nHeight, nRowAlignBytes);
    }

    ImageT(int nWidth, int nHeight, T* pBits, size_t nStride)
    {
        AttachData(nWidth, nHeight, SizeofT(), nBands, (unsigned char*)pBits, nStride);
    }

    void Create(int nWidth, int nHeight, unsigned int nRowAlignBytes = 4)
    {
        Image::Create(nWidth, nHeight, SizeofT(), nBands, nRowAlignBytes);
    }

    T* Ptr()
    {
        return (T*)BytePtr();
    }
    T* Ptr(int y)
    {
        return (T*)(BytePtr() + y * StrideBytes());
    }
    T* Ptr(int x, int y)
    {
        return (T*)(BytePtr() + y * StrideBytes() + x * PixelBytes());
    }

    const T* Ptr() const
    {
        return (const T*)BytePtr();
    }
    const T* Ptr(int y) const
    {
        return (const T*)(BytePtr() + y * StrideBytes());
    }
    const T* Ptr(int x, int y) const
    {
        return (const T*)(BytePtr() + y * StrideBytes() + x * PixelBytes());
    }

    T& operator()(int x, int y)
    {
        return *Ptr(x, y);
    }

    const T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }

    size_t Stride() const
    {
        return StrideBytes() / SizeofT();
    }

    void Clear()
    {
        Image::Clear();
    }

    void Clear(const T& t)
    {
        for(int y = 0; y < Height(); y++)
        {
            int cx = Width() * Bands();
            T* p = Ptr(y);

            for(int x = 0; x < cx; x++)
                p[x] = t;
        }
    }

protected:
    static size_t SizeofT()
    {
        return (size_t)((T*)0 + 1);
    }
};

template <typename T, unsigned int nBands = 1>
class ImageRef
{
public:
    ImageRef() {}
    ImageRef(int nWidth, int nHeight) : m_p(new ImageT<T, nBands>(nWidth, nHeight))
    {
        m_p->Clear();
    }
    ImageRef(const ImageRef& rhs) : m_p(rhs.m_p) {}

    T* Ptr() const
    {
        return m_p->Ptr();
    }
    T* Ptr(int y) const
    {
        return m_p->Ptr(y);
    }
    T* Ptr(int x, int y) const
    {
        return m_p->Ptr(x, y);
    }

    ImageT<T, nBands>& Get() const
    {
        return *m_p;
    }

    int Width() const
    {
        return m_p->Width();
    }
    int Height() const
    {
        return m_p->Height();
    }
    unsigned int Bands() const
    {
        return nBands;
    }
    size_t StrideBytes() const
    {
        return m_p->StrideBytes();
    }
    void Clear() const
    {
        return m_p->Clear();
    }
    void Clear(const T& t) const
    {
        m_p->Clear(t);
    }
    std::shared_ptr<ImageT<T, nBands>> Ref() const
    {
        return m_p;
    }

    T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }
    operator bool() const
    {
        return static_cast<bool>(m_p);
    }
protected:
    std::shared_ptr<ImageT<T, nBands>> m_p;
};

template <typename T, unsigned int nBands = 1>
class ImageRefC
{
public:
    ImageRefC() {}
    ImageRefC(const ImageRefC& rhs) : m_p(rhs.m_p) {}
    ImageRefC(const ImageRef<T, nBands>& rhs) : m_p(rhs.Ref()) {}

    const T* Ptr() const
    {
        return m_p->Ptr();
    }
    const T* Ptr(int y) const
    {
        return m_p->Ptr(y);
    }
    const T* Ptr(int x, int y) const
    {
        return m_p->Ptr(x, y);
    }

    int Width() const
    {
        return m_p->Width();
    }
    int Height() const
    {
        return m_p->Height();
    }
    unsigned int Bands() const
    {
        return nBands;
    }
    size_t StrideBytes() const
    {
        return m_p->StrideBytes();
    }

    const T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }
    operator bool() const
    {
        return m_p ? true : false;
    }
protected:
    std::shared_ptr<const ImageT<T, nBands>> m_p;
};

#endif // H_RTF_IMAGE_H
