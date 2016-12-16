/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Trees.h
 * Implements a simple class for representing binary regression trees. The
 * implementation is array-based and most efficient for dense trees.
 *
 */

#ifndef H_RTF_TREES_H
#define H_RTF_TREES_H

#include <vector>
#include <stdexcept>

#include "Array.h"
#include "Rect.h"

/*
  A tree consists of a node structure, a feature test object
  for each internal node, and additional data from training
  stored at each node.
*/
template <class TFeature, class TTrainingData>
struct NodeData
{
    NodeData() {}
    NodeData(const TTrainingData& d) : data(d) {}
    NodeData(const TFeature& f, const TTrainingData& d) : feature(f), data(d) {}
    NodeData(const NodeData& rhs) : feature(rhs.feature), data(rhs.data) {}

    TFeature feature;
    TTrainingData data;
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class Tree
{
public:

    typedef NodeData<TFeature, TTrainingData> TNodeData;
    typedef Tree<TFeature, TTrainingData, TAllocator> TTree;

    typedef TNodeData  value_type;
    typedef TNodeData* pointer_type;
    typedef TNodeData& reference_type;

    class iterator_base
    {
    public:
        iterator_base() : node(-1), tree(nullptr) {}
        iterator_base(int n, TTree* t) : node(n), tree(t) {}

        reference_type operator*() const
        {
            return (*tree)[node];
        }

        pointer_type operator->() const
        {
            return &((*tree)[node]);
        }

        operator bool() const
        {
            return node != -1;
        }

        bool operator ==(const iterator_base& rhs) const
        {
            return (node == rhs.node);
        }

        bool operator !=(const iterator_base& rhs) const
        {
            return !operator ==(rhs);
        }

        int number_of_children() const
        {
            return tree->number_of_children(node);
        }

        int right_child() const
        {
            return tree->right_child(node);
        }

        int left_child() const
        {
            return tree->left_child(node);
        }

        const TFeature& feature() const
        {
            return (*tree)[node].feature;
        }

        bool used() const
        {
            return tree->used[node];
        }

        bool valid() const
        {
            return (node >= 0) && (node < tree->used.size());
        }

        int node;
        TTree* tree;
    };

    class test_time_iterator : public iterator_base
    {
    public:
        typedef iterator_base base;

        test_time_iterator() : m_x(-1), m_y(-1), m_prep(nullptr) {}
        test_time_iterator(int n, TTree* t, int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets)
            : iterator_base(n, t), m_x(x), m_y(y), m_prep(&preProcessed), m_offsets(offsets) {}

        test_time_iterator& operator++()
        {
            if( base::number_of_children() > 0 )
            {
                const auto branch = base::feature()(m_x, m_y, *m_prep, m_offsets);
                base::node = branch ? base::right_child() : base::left_child();
            }
            else
            {
                base::node = -1;
            }

            return *this;
        }

    protected:
        int m_x, m_y;
        const typename TFeature::PreProcessType* m_prep;
        VecCRef<Vector2D<int> > m_offsets;
    };

    class breadth_first_iterator : public iterator_base
    {
    public:
        typedef iterator_base base;

        breadth_first_iterator() : iterator_base() {}
        breadth_first_iterator(int n, TTree* t) : iterator_base(n, t)
        {
            seek_used();
        }

        breadth_first_iterator& operator++()
        {
            ++base::node;
            seek_used();
            return *this;
        }

    private:
        void seek_used()
        {
            // Skip any unused slots
            while( base::valid() && (! base::used()) )
                ++base::node;

            if( !base::valid() )
                base::node = -1;
        }
    };

    class leaf_iterator : public iterator_base
    {
    public:
        typedef iterator_base base;

        leaf_iterator() : iterator_base() {}
        leaf_iterator(int n, TTree* t) : iterator_base(n, t)
        {
            seek_used();
        }

        leaf_iterator& operator++()
        {
            ++base::node;
            seek_used();
            return *this;
        }

    private:
        void seek_used()
        {
            // Skip any unused slots and internal nodes
            while( base::valid() && ( (!base::used()) || (base::number_of_children() != 0) ))
                ++base::node;

            if( !base::valid() )
                base::node = -1;
        }
    };

    test_time_iterator begin_test(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        if ( size() > 0 )
            return test_time_iterator(0, const_cast<TTree*>(this), x, y, preProcessed, offsets);
        else
            return end_test();
    }

    test_time_iterator end_test() const
    {
        return test_time_iterator();
    }

    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        auto it = begin_test(x, y, preProcessed, offsets);

        while( it.number_of_children() > 0 )
            ++it;

        return it;
    }

    breadth_first_iterator begin_breadth_first() const
    {
        if( size() > 0 )
            return breadth_first_iterator(0, const_cast<TTree*>(this));
        else
            return end_breadth_first();
    }

    breadth_first_iterator end_breadth_first() const
    {
        return breadth_first_iterator();
    }

    leaf_iterator begin_leaf() const
    {
        if( size() > 0 )
            return leaf_iterator(0, const_cast<TTree*>(this));
        else
            return leaf_iterator();
    }

    leaf_iterator end_leaf() const
    {
        return leaf_iterator();
    }

    iterator_base set_head(const value_type& value)
    {
        ensure_capacity(0);
        nodes[0]   = value;
        used[0]    = true;
        num_nodes += 1;

        return iterator_base(0, this);
    }

    template<typename iter> iter append_child(iter it, const value_type& x) const
    {
        int idx = -1;

        if( it.left_child() == -1 )
            idx = left_child_index(it.node);
        else if ( it.right_child() == -1 )
            idx = right_child_index(it.node);
        else
            throw std::runtime_error("Attempt to add more than two children to a node.");

        ensure_capacity(idx);
        nodes[idx] = x;
        used[idx]  = true;
        num_nodes += 1;

        iter ret = it;
        ret.node = idx;
        return ret;
    }

    template<typename iter> iter append_child(iter it) const
    {
        return append_child(it, value_type());
    }

    static int depth(const iterator_base& i)
    {
        return static_cast<int>(std::log(i.node + 1.4)/std::log(2));
    }

    size_t size() const
    {
        return num_nodes;
    }

    value_type& operator[](size_t idx)
    {
        return nodes[idx];
    }

    const value_type& operator[](size_t idx) const
    {
        return nodes[idx];
    }

    Tree() : num_nodes(0) {}

private:

    int left_child(int parent_index) const
    {
        if( parent_index == -1 )
            return -1;
        else if ( left_child_index(parent_index) >= used.size() )
            return -1;
        else if ( ! used[left_child_index(parent_index)] )
            return -1;
        else
            return left_child_index(parent_index);
    }

    int right_child(int parent_index) const
    {
        if( parent_index == -1 )
            return -1;
        else if ( right_child_index(parent_index) >= used.size() )
            return -1;
        else if ( ! used[right_child_index(parent_index)] )
            return -1;
        else
            return right_child_index(parent_index);
    }

    static int left_child_index(int parent_index)
    {
        return 2 * parent_index + 1;
    }

    static int right_child_index(int parent_index)
    {
        return 2 * parent_index + 2;
    }

    int number_of_children( int node_index ) const
    {
        return static_cast<int>(left_child(node_index) != -1) + static_cast<int>(right_child(node_index) != -1);
    }

    void ensure_capacity(int node_index) const
    {
        nodes.resize(node_index+1);
        used.resize(node_index+1);
    }

    mutable int num_nodes;
    mutable std::vector<TNodeData, TAllocator> nodes;
    mutable std::vector<bool, TAllocator> used;
};


template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class TreeRef
{
public:
    typedef Tree<TFeature, TTrainingData, TAllocator> TBase;

    TreeRef() : m_p(new TBase()) {}
    TreeRef(const TreeRef& rhs) : m_p(rhs.m_p) {}
    TreeRef& operator =(const TreeRef& rhs)
    {
        m_p = rhs.m_p;
        return *this;
    }
    std::shared_ptr<TBase> Ref() const
    {
        return m_p;
    }

    typedef typename TBase::value_type value_type;

    typedef typename TBase::iterator_base iterator_base;
    typedef typename TBase::leaf_iterator leaf_iterator;
    typedef typename TBase::test_time_iterator test_time_iterator;
    typedef typename TBase::breadth_first_iterator breadth_first_iterator;

    static int depth(const iterator_base& i)
    {
        return TBase::depth(i);
    }

    void set_head(const value_type& value)
    {
        m_p->set_head(value);
    }

    breadth_first_iterator begin_breadth_first() const
    {
        return m_p->begin_breadth_first();
    }

    breadth_first_iterator end_breadth_first() const
    {
        return m_p->end_breadth_first();
    }

    template<typename iter> iter append_child(iter it, const value_type& x) const
    {
        return m_p->append_child(it, x);
    }

    template<typename iter> iter append_child(iter it) const
    {
        return m_p->append_child(it);
    }

    leaf_iterator begin_leaf() const
    {
        return m_p->begin_leaf();
    }

    leaf_iterator end_leaf() const
    {
        return m_p->end_leaf();
    }

    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->goto_leaf(x, y, preProcessed, offsets);
    }

    size_t size() const
    {
        return m_p->size();
    }

private:
    std::shared_ptr<TBase> m_p;
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class TreeCRef
{
public:
    typedef Tree<TFeature, TTrainingData, TAllocator> TBase;
    typedef typename TBase::value_type value_type;
    typedef typename TBase::iterator_base iterator_base;
    typedef typename TBase::test_time_iterator test_time_iterator;
    typedef typename TBase::breadth_first_iterator breadth_first_iterator;

    TreeCRef() : m_p(new TBase()) {}
    TreeCRef(const TreeRef<TFeature, TTrainingData, TAllocator>& rhs) : m_p(rhs.Ref()) {}
    TreeCRef(const TreeCRef& rhs) : m_p(rhs.m_p) {}
    TreeCRef& operator =(const TreeCRef& rhs)
    {
        m_p = rhs.m_p;
        return *this;
    }

    breadth_first_iterator begin_breadth_first() const
    {
        return m_p->begin_breadth_first();
    }
    breadth_first_iterator end_breadth_first() const
    {
        return m_p->end_breadth_first();
    }
    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->goto_leaf(x, y, preProcessed, offsets);
    }
    size_t size() const
    {
        return m_p->size();
    }
    const TTrainingData& GetLeafData(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int> >& offsets) const
    {
        return goto_leaf(x, y, prep, offsets)->data;
    }

private:
    std::shared_ptr<const TBase> m_p;
};

class TreeTable
{
public:
    virtual ~TreeTable() {}
};

template <typename TFeature, typename TTrainingData>
class TreeTableT : public TreeTable
{
public:
    TreeTableT() {}
    virtual ~TreeTableT() {}

    template <typename T2, typename TOp>
    void Fill(const TreeCRef<TFeature, T2>& root, TOp op)
    {
        // Pre-allocate table
        m_entries.reserve(root.size());
        int nThis = 0, nNext = 1;

        for(auto i = root.begin_breadth_first(); i != root.end_breadth_first(); ++i)
        {
            bool bLeaf = i.number_of_children() == 0;
            Entry entry;
            entry.feature = i->feature;
            entry.data = op(i);
            entry.entrySkip = bLeaf ? -1 : nNext - nThis;
            m_entries.push_back(entry);
            nNext += i.number_of_children();
            nThis++;
        }
    }

    // Returns the index of the leaf node, using a breadth-first numbering
    TTrainingData GetLeafData(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        const Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            bool b = pEntry->feature(x, y, preProcessed, offsets);
            pEntry += pEntry->entrySkip + b;
        }

        return pEntry->data;
    }

protected:
    struct Entry
    {
        Entry() {}
        Entry(const Entry& rhs) : feature(rhs.feature), data(rhs.data), entrySkip(rhs.entrySkip) {}

        TFeature feature;
        TTrainingData data;
        int entrySkip; ///< Number of entries to skip forward for a false test, -1 designates leaf node
    };
    std::vector<Entry> m_entries;
};

#endif // H_RTF_TREES_H
