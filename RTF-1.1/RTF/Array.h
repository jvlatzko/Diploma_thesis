/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Array.h
 * Implements a reference-counted wrapper arround std::vector.
 *
 */

#ifndef H_RTF_ARRAY_H
#define H_RTF_ARRAY_H

#include <vector>
#include <memory>

template <typename T> class VecCRef;

template <typename T>
class VecRef
{
    friend class VecCRef<T>;

public:
    VecRef() : m_p(new std::vector<T>()) {}
    VecRef(size_t size) : m_p(new std::vector<T>(size)) {}
    VecRef(size_t size, const T& t) : m_p(new std::vector<T>(size, t)) {}
    VecRef(const VecRef<T>& rhs) : m_p(rhs.m_p) {}
    VecRef(const std::vector<T>& rhs) : m_p(new std::vector<T>(rhs)) {}

    explicit VecRef(const VecCRef<T>& rhs) // Element-wise copy
        : m_p(new std::vector<T>(rhs.size()))
    {
        for(size_t i = 0; i < rhs.size(); i++)
            operator[](i) = rhs[i];
    }

    typedef typename std::vector<T>::const_iterator const_iterator;

    size_t size() const
    {
        return m_p->size();
    }
    void push_back(const T& t)
    {
        m_p->push_back(t);
    }
    void resize(size_t size)
    {
        m_p->resize(size);
    }
    T& operator[](size_t index) const
    {
        return m_p->operator[](index);
    }

    bool empty() const
    {
        return m_p->empty();
    }
    const_iterator begin() const
    {
        return m_p->begin();
    }
    const_iterator end() const
    {
        return m_p->end();
    }

    operator std::vector<T>&()
    {
        return *m_p;
    }

    void reserve(size_t size)
    {
        m_p->reserve(size);
    }
protected:
    std::shared_ptr<std::vector<T>> m_p;
};

template <typename T>
class VecCRef
{
public:
    VecCRef() : m_p(new std::vector<T>()) {}
    VecCRef(const VecCRef<T>& rhs) : m_p(rhs.m_p) {}
    VecCRef(const VecRef<T>& rhs) : m_p(rhs.m_p) {}
    VecCRef(const std::vector<T>& rhs) : m_p(new std::vector<T>(rhs)) {}

    typedef typename std::vector<T>::const_iterator const_iterator;

    size_t size() const
    {
        return m_p->size();
    }
    const T& operator[](size_t index) const
    {
        return m_p->operator[](index);
    }
    bool empty() const
    {
        return m_p->empty();
    }
    const_iterator begin() const
    {
        return m_p->begin();
    }
    const_iterator end() const
    {
        return m_p->end();
    }
    operator const std::vector<T>&() const
    {
        return *m_p;
    }
    const std::vector<T>& operator *() const
    {
        return *m_p;
    }

protected:
    std::shared_ptr<const std::vector<T>> m_p;
};

#endif // H_RTF_ARRAY_H
