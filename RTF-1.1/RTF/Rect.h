/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Rect.h
 * Implements simple functionality related to rectangles.
 *
 */

#ifndef H_RTF_RECT_H
#define H_RTF_RECT_H

// Implements a 2D vector class and a rectangle class
// The rectangle supports trunctuation and deflation by vectors
// and queries whether a vector is contained in the rectangle

template <typename T>
struct Vector2D
{
    Vector2D() : x((T)0), y((T)0) {}
    Vector2D(T xx, T yy) : x(xx), y(yy) {}
    T x, y;

};

template <typename T>
inline Vector2D<T> operator +(const Vector2D<T>& lhs, const Vector2D<T>& rhs)
{
    return Vector2D<T>(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <typename T>
inline Vector2D<T> operator -(const Vector2D<T>& lhs, const Vector2D<T>& rhs)
{
    return Vector2D<T>(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <typename T>
inline bool operator ==(const Vector2D<T>& lhs, const Vector2D<T>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <typename T>
struct Rect
{
    Rect() : left(0), top(0), right(0), bottom(0) {}
    Rect(T l, T t, T r, T b) : left(l), top(t), right(r), bottom(b) {}

    Rect& operator |=(const Vector2D<T>& pt)
    {
        left = std::min(left, pt.x);
        right = std::max(right, pt.x);
        top = std::min(top, pt.y);
        bottom = std::max(bottom, pt.y);
        return *this;
    }

    Rect<T> DeflateRect(const Rect<T>& rhs) const
    {
        return Rect<T>(left - rhs.left, top - rhs.top,
                       right - rhs.right, bottom - rhs.bottom);
    }

    T Width() const
    {
        return right - left;
    }
    T Height() const
    {
        return bottom - top;
    }

    bool PtInRect(const Vector2D<T>& pt) const
    {
        return pt.x >= left && pt.x < right &&
               pt.y >= top && pt.y < bottom;
    }

    T left, top, right, bottom;
};

#endif // H_RTF_RECT_H
