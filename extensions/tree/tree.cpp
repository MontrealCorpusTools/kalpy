
#include "tree/pybind_tree.h"
#include "tree/build-tree-questions.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "tree/context-dep.h"
#include "tree/event-map.h"
#include "tree/tree-renderer.h"
#include "hmm/hmm-topology.h"

using namespace kaldi;

class PyEventMap : public EventMap {
public:
    //Inherit the constructors
    using EventMap::EventMap;

    //Trampoline (need one for each virtual function)
    void Write(std::ostream &os, bool binary) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            Write,          //Name of function in C++ (must match Python name) (fn)
            os, binary      //Argument(s) (...)
        );
    }

    EventAnswerType MaxResult() const override {
        PYBIND11_OVERRIDE_PURE(
            EventAnswerType, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            MaxResult         //Name of function in C++ (must match Python name) (fn)
        );
    }

    EventMap *Prune() const override {
        PYBIND11_OVERRIDE_PURE(
            EventMap*, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            Prune          //Name of function in C++ (must match Python name) (fn)
        );
    }

    EventMap *Copy(const std::vector<EventMap*> &new_leaves) const override {
        PYBIND11_OVERRIDE_PURE(
            EventMap*, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
            new_leaves      //Argument(s) (...)
        );
    }

    EventMap *MapValues(const unordered_set<EventKeyType> &keys_to_map,
      const unordered_map<EventValueType,EventValueType> &value_map) const override {
        PYBIND11_OVERRIDE_PURE(
            EventMap*, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            MapValues          //Name of function in C++ (must match Python name) (fn)
            keys_to_map, value_map      //Argument(s) (...)
        );
    }

    void GetChildren(std::vector<EventMap*> *out) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            GetChildren          //Name of function in C++ (must match Python name) (fn)
            out      //Argument(s) (...)
        );
    }

    void MultiMap(const EventType &event, std::vector<EventAnswerType> *ans) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            MultiMap          //Name of function in C++ (must match Python name) (fn)
            event, ans      //Argument(s) (...)
        );
    }

    bool Map(const EventType &event, EventAnswerType *ans) const override {
        PYBIND11_OVERRIDE_PURE(
            bool, //Return type (ret_type)
            EventMap,      //Parent class (cname)
            Map          //Name of function in C++ (must match Python name) (fn)
            event, ans      //Argument(s) (...)
        );
    }
};


void pybind_event_map(py::module& m) {
  auto event_map = py::class_<EventMap, PyEventMap /*<--- trampoline*/>(m, "EventMap",
      "A class that is capable of representing a generic mapping from "
      "EventType (which is a vector of (key, value) pairs) to "
      "EventAnswerType which is just an integer.  See //ref tree_internals "
      "for overview.");
  event_map.def(py::init<>());
  event_map
      .def_static("Read", &EventMap::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def_static("Write", py::overload_cast<std::ostream &, bool, EventMap *>(&EventMap::Write),
                    py::arg("os"), py::arg("binary"), py::arg("emap"),
      py::call_guard<py::gil_scoped_release>());
  {
    using PyClass = ConstantEventMap;

    auto const_event_map = py::class_<ConstantEventMap, EventMap>(
        m, "ConstantEventMap");
  }
  {
    using PyClass = TableEventMap;

    auto table_event_map = py::class_<TableEventMap, EventMap>(
        m, "TableEventMap");
  }
  {
    using PyClass = SplitEventMap;

    auto split_event_map = py::class_<SplitEventMap, EventMap>(
        m, "SplitEventMap");
  }
  m.def("GetTreeStructure",
        (void (*)(const EventMap &,
                      int32 *,
                      std::vector<int32> *)) &GetTreeStructure,
        "This function gets the tree structure of the EventMap \"map\" in a convenient form. "
        "If \"map\" corresponds to a tree structure (not necessarily binary) with leaves "
        "uniquely numbered from 0 to num_leaves-1, then the function will return true, "
        "output \"num_leaves\", and set \"parent\" to a vector of size equal to the number of "
        "nodes in the tree (nonleaf and leaf), where each index corresponds to a node "
        "and the leaf indices correspond to the values returned by the EventMap from "
        "that leaf; for an index i, parent[i] equals the parent of that node in the tree "
        "structure, where parent[i] > i, except for the last (root) node where parent[i] == i. "
        "If the EventMap does not have this structure (e.g. if multiple different leaf nodes share "
        "the same number), then it will return false.",
        py::arg("map"),
        py::arg("num_leaves"),
        py::arg("parents"));

}

void pybind_build_tree_questions(py::module& m) {

  {
    using PyClass = QuestionsForKey;

    auto questions_for_key = py::class_<PyClass>(
        m, "QuestionsForKey",
        "QuestionsForKey is a class used to define the questions for a key, "
        "and also options that allow us to refine the question during tree-building "
        "(i.e. make a question specific to the location in the tree). "
        "The Questions class handles aggregating these options for a set "
        "of different keys.");

    questions_for_key.def(py::init<>())
      .def(py::init<int32>(), py::arg("num_iters") = 5)
        .def("Check", &PyClass::Check)
        .def("Read", &PyClass::Read,
          py::arg("is"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Write", &PyClass::Write,
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }

  {
    using PyClass = Questions;

    auto questions = py::class_<PyClass>(
        m, "Questions",
        "This class defines, for each EventKeyType, a set of initial questions that "
        "it tries and also a number of iterations for which to refine the questions to increase "
        "likelihood. It is perhaps a bit more than an options class, as it contains the "
        "actual questions.");

    questions.def(py::init<>())
        .def("GetQuestionsOf", &PyClass::GetQuestionsOf, py::arg("key"))
        .def("SetQuestionsOf", &PyClass::SetQuestionsOf, py::arg("key"), py::arg("options_of_key"))
        .def("GetKeysWithQuestions", &PyClass::GetKeysWithQuestions, py::arg("keys_out"))
        .def("HasQuestionsForKey", &PyClass::HasQuestionsForKey, py::arg("key"))
        .def("InitRand", &PyClass::InitRand, py::arg("stats"), py::arg("num_quest"),
                                    py::arg("num_iters_refine"), py::arg("all_keys_type"))
        .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }
}

void pybind_build_tree_utils(py::module& m) {

  m.def("DeleteBuildTreeStats",
        &DeleteBuildTreeStats,
        "This frees the Clusterable* pointers in \"stats\", where non-NULL, and sets them to NULL. "
        "Does not delete the pointer \"stats\" itself.",
        py::arg("stats"));
  m.def("WriteBuildTreeStats",
        &WriteBuildTreeStats,
        "Writes BuildTreeStats object.  This works even if pointers are NULL.",
        py::arg("os"),
        py::arg("binary"),
        py::arg("stats"));
  m.def("ReadBuildTreeStats",
        &ReadBuildTreeStats,
        "Reads BuildTreeStats object.  The \"example\" argument must be of the same "
        "type as the stats on disk, and is needed for access to the correct \"Read\" "
        "function.  It was organized this way for easier extensibility (so adding new "
        "Clusterable derived classes isn't painful)",
        py::arg("is"),
        py::arg("binary"),
        py::arg("example"),
        py::arg("stats"));
  m.def("PossibleValues",
        &PossibleValues,
        "Convenience function e.g. to work out possible values of the phones from just the stats. "
        "Returns true if key was always defined inside the stats. "
        "May be used with and == NULL to find out of key was always defined.",
        py::arg("key"),
        py::arg("stats"),
        py::arg("ans"));
  m.def("SplitStatsByMap",
        &SplitStatsByMap,
        "Splits stats according to the EventMap, indexing them at output by the "
        "leaf type.   A utility function.  NOTE-- pointers in stats_out point to "
        "the same memory location as those in stats.  No copying of Clusterable* "
        "objects happens.  Will add to stats in stats_out if non-empty at input. "
        "This function may increase the size of vector stats_out as necessary "
        "to accommodate stats, but will never decrease the size.",
        py::arg("stats_in"),
        py::arg("e"),
        py::arg("stats_out"));
  m.def("SplitStatsByKey",
        &SplitStatsByKey,
        "SplitStatsByKey splits stats up according to the value of a particular key, "
        "which must be always defined and nonnegative.  Like MapStats.  Pointers to "
        "Clusterable* in stats_out are not newly allocated-- they are the same as the "
        "ones in stats_in.  Generally they will still be owned at stats_in (user can "
        "decide where to allocate ownership).",
        py::arg("stats_in"),
        py::arg("key"),
        py::arg("stats_out"));
  m.def("ConvertStats",
        &ConvertStats,
        "Converts stats from a given context-window (N) and central-position (P) to a "
        "different N and P, by possibly reducing context.  This function does a job "
        "that's quite specific to the \"normal\" stats format we use.  See //ref "
        "tree_window for background.  This function may delete some keys and change "
        "others, depending on the N and P values.  It expects that at input, all keys "
        "will either be -1 or lie between 0 and oldN-1.  At output, keys will be "
        "either -1 or between 0 and newN-1. "
        "Returns false if we could not convert the stats (e.g. because newN is larger "
        "than oldN).",
        py::arg("oldN"),
        py::arg("oldP"),
        py::arg("newN"),
        py::arg("newP"),
        py::arg("stats"));
  m.def("FilterStatsByKey",
        &FilterStatsByKey,
        "FilterStatsByKey filters the stats according the value of a specified key. "
        "If include_if_present == true, it only outputs the stats whose key is in "
        "\"values\"; otherwise it only outputs the stats whose key is not in \"values\". "
        "At input, \"values\" must be sorted and unique, and all stats in \"stats_in\" "
        "must have \"key\" defined.  At output, pointers to Clusterable* in stats_out "
        "are not newly allocated-- they are the same as the ones in stats_in.",
        py::arg("stats_in"),
        py::arg("key"),
        py::arg("values"),
        py::arg("include_if_present"),
        py::arg("stats_out"));
  m.def("SumStats",
        &SumStats,
        "Sums stats, or returns NULL stats_in has no non-NULL stats. "
        "Stats are newly allocated, owned by caller.",
        py::arg("stats_in"));
  m.def("SumNormalizer",
        &SumNormalizer,
        "Sums the normalizer [typically, data-count] over the stats.",
        py::arg("stats_in"));
  m.def("SumObjf",
        &SumObjf,
        "Sums the objective function over the stats.",
        py::arg("stats_in"));
  m.def("SumStatsVec",
        &SumStatsVec,
        "Sum a vector of stats.  Leaves NULL as pointer if no stats available. "
        "The pointers in stats_out are owned by caller.  At output, there may be "
        "NULLs in the vector stats_out.",
        py::arg("stats_in"),
        py::arg("stats_out"));
  m.def("ObjfGivenMap",
        &ObjfGivenMap,
        "Cluster the stats given the event map return the total objf given those clusters.",
        py::arg("stats_in"),
        py::arg("e"));
  m.def("FindAllKeys",
        &FindAllKeys,
        "FindAllKeys puts in *keys the (sorted, unique) list of all key identities in the stats. "
        "If type == kAllKeysInsistIdentical, it will insist that this set of keys is the same for all the "
        "  stats (else exception is thrown). "
        "if type == kAllKeysIntersection, it will return the smallest common set of keys present in "
        "  the set of stats "
        "if type== kAllKeysUnion (currently probably not so useful since maps will return \"undefined\" "
        "  if key is not present), it will return the union of all the keys present in the stats.",
        py::arg("stats"),
        py::arg("keys_type"),
        py::arg("keys"));
  m.def("TrivialTree",
        &TrivialTree,
        "Returns a tree with just one node.  Used @ start of tree-building process. "
        "Not really used in current recipes.",
        py::arg("num_leaves"));
  m.def("DoTableSplit",
        &DoTableSplit,
        "DoTableSplit does a complete split on this key (e.g. might correspond to central phone "
        "(key = P-1), or HMM-state position (key == kPdfClass == -1).  Stats used to work out possible "
        "values of the event. \"num_leaves\" is used to allocate new leaves.   All stats must have "
        "this key defined, or this function will crash.",
        py::arg("orig"),
        py::arg("key"),
        py::arg("stats"),
        py::arg("num_leaves"));
  m.def("DoTableSplitMultiple",
        &DoTableSplitMultiple,
        "DoTableSplitMultiple does a complete split on all the keys, in order from keys[0], "
        "keys[1] "
        "and so on.  The stats are used to work out possible values corresponding to the key. "
        "\"num_leaves\" is used to allocate new leaves.   All stats must have "
        "the keys defined, or this function will crash. "
        "Returns a newly allocated event map.",
        py::arg("orig"),
        py::arg("keys"),
        py::arg("stats"),
        py::arg("num_leaves"));
  m.def("ClusterEventMapGetMapping",
        &ClusterEventMapGetMapping,
        "\"ClusterEventMapGetMapping\" clusters the leaves of the EventMap, with \"thresh\" a delta-likelihood "
        "threshold to control how many leaves we combine (might be the same as the delta-like "
        "threshold used in splitting. "
        "The function returns the #leaves we combined.  The same leaf-ids of the leaves being clustered "
        "will be used for the clustered leaves (but other than that there is no special rule which "
        "leaf-ids should be used at output). "
        "It outputs the mapping for leaves, in \"mapping\", which may be empty at the start "
        "but may also contain mappings for other parts of the tree, which must contain "
        "disjoint leaves from this part.  This is so that Cluster can "
        "be called multiple times for sub-parts of the tree (with disjoint sets of leaves), "
        "e.g. if we want to avoid sharing across phones.  Afterwards you can use Copy function "
        "of EventMap to apply the mapping, i.e. call e_in.Copy(mapping) to get the new map. "
        "Note that the application of Cluster creates gaps in the leaves.  You should then "
        "call RenumberEventMap(e_in.Copy(mapping), num_leaves). "
        "*If you only want to cluster a subset of the leaves (e.g. just non-silence, or just "
        "a particular phone, do this by providing a set of \"stats\" that correspond to just "
        "this subset of leaves*.  Leaves with no stats will not be clustered. "
        "See build-tree.cc for an example of usage.",
        py::arg("e_in"),
        py::arg("stats"),
        py::arg("thresh"),
        py::arg("mapping"));
  m.def("ClusterEventMap",
        &ClusterEventMap,
        "This is as ClusterEventMapGetMapping but a more convenient interface "
        "that exposes less of the internals.  It uses a bottom-up clustering to "
        "combine the leaves, until the log-likelihood decrease from combinging two "
        "leaves exceeds the threshold.",
        py::arg("e_in"),
        py::arg("stats"),
        py::arg("thresh"),
        py::arg("num_removed"));
  m.def("ClusterEventMapRestrictedByKeys",
        &ClusterEventMapRestrictedByKeys,
        "This is as ClusterEventMap, but first splits the stats on the keys specified "
        "in \"keys\" (e.g. typically keys = [ -1, P ]), and only clusters within the "
        "classes defined by that splitting. "
        "Note-- leaves will be non-consecutive at output, use RenumberEventMap.",
        py::arg("e_in"),
        py::arg("stats"),
        py::arg("thresh"),
        py::arg("keys"),
        py::arg("num_removed"));
  m.def("ClusterEventMapRestrictedByMap",
        &ClusterEventMapRestrictedByMap,
        "This version of ClusterEventMapRestricted restricts the clustering to only "
        "allow things that \"e_restrict\" maps to the same value to be clustered "
        "together.",
        py::arg("e_in"),
        py::arg("stats"),
        py::arg("thresh"),
        py::arg("e_restrict"),
        py::arg("num_removed"));
  m.def("ClusterEventMapToNClustersRestrictedByMap",
        &ClusterEventMapToNClustersRestrictedByMap,
        "This version of ClusterEventMapRestrictedByMap clusters to get a "
        "specific number of clusters as specified by 'num_clusters'",
        py::arg("e_in"),
        py::arg("stats"),
        py::arg("num_clusters"),
        py::arg("e_restrict"),
        py::arg("num_removed"));
  m.def("RenumberEventMap",
        &RenumberEventMap,
        "RenumberEventMap [intended to be used after calling ClusterEventMap] renumbers "
        "an EventMap so its leaves are consecutive. "
        "It puts the number of leaves in *num_leaves.  If later you need the mapping of "
        "the leaves, modify the function and add a new argument.",
        py::arg("e_in"),
        py::arg("num_leaves"));
  m.def("MapEventMapLeaves",
        &MapEventMapLeaves,
        "This function remaps the event-map leaves using this mapping, "
        "indexed by the number at leaf.",
        py::arg("e_in"),
        py::arg("mapping"));
  m.def("ShareEventMapLeaves",
        &ShareEventMapLeaves,
        "ShareEventMapLeaves performs a quite specific function that allows us to "
        "generate trees where, for a certain list of phones, and for all states in "
        "the phone, all the pdf's are shared. "
        "Each element of \"values\" contains a list of phones (may be just one phone), "
        "all states of which we want shared together).  Typically at input, \"key\" will "
        "equal P, the central-phone position, and \"values\" will contain just one "
        "list containing the silence phone. "
        "This function renumbers the event map leaves after doing the sharing, to "
        "make the event-map leaves contiguous.",
        py::arg("e_in"),
        py::arg("key"),
        py::arg("values"),
        py::arg("num_leaves"));
  m.def("SplitDecisionTree",
        &SplitDecisionTree,
        "Does a decision-tree split at the leaves of an EventMap. "
        "@param orig [in] The EventMap whose leaves we want to split. [may be either a trivial or a "
        "          non-trivial one]. "
        "@param stats [in] The statistics for splitting the tree; if you do not want a particular "
        "         subset of leaves to be split, make sure the stats corresponding to those leaves "
        "         are not present in \"stats\". "
        "@param qcfg [in] Configuration class that contains initial questions (e.g. sets of phones) "
        "         for each key and says whether to refine these questions during tree building. "
        "@param thresh [in] A log-likelihood threshold (e.g. 300) that can be used to "
        "          limit the number of leaves; you can use zero and set max_leaves instead. "
        "@param max_leaves [in] Will stop leaves being split after they reach this number. "
        "@param num_leaves [in,out] A pointer used to allocate leaves; always corresponds to the "
        "            current number of leaves (is incremented when this is increased). "
        "@param objf_impr_out [out] If non-NULL, will be set to the objective improvement due to splitting "
        "          (not normalized by the number of frames). "
        "@param smallest_split_change_out If non-NULL, will be set to the smallest objective-function "
        "        improvement that we got from splitting any leaf; useful to provide a threshold "
        "        for ClusterEventMap. "
        "@return The EventMap after splitting is returned; pointer is owned by caller.",
        py::arg("orig"),
        py::arg("stats"),
        py::arg("qcfg"),
        py::arg("thresh"),
        py::arg("max_leaves"),
        py::arg("num_leaves"),
        py::arg("objf_impr_out"),
        py::arg("smallest_split_change_out"));
  m.def("FindBestSplitForKey",
        &FindBestSplitForKey,
        "FindBestSplitForKey is a function used in DoDecisionTreeSplit. "
        "It finds the best split for this key, given these stats. "
        "It will return 0 if the key was not always defined for the stats.",
        py::arg("stats"),
        py::arg("qcfg"),
        py::arg("key"),
        py::arg("yes_set"));
  m.def("GetStubMap",
        &GetStubMap,
        "GetStubMap is used in tree-building functions to get the initial "
        "to-states map, before the decision-tree-building process.  It creates "
        "a simple map that splits on groups of phones.  For the set of phones in "
        "phone_sets[i] it creates either: if share_roots[i] == true, a single "
        "leaf node, or if share_roots[i] == false, separate root nodes for "
        "each HMM-position (it goes up to the highest position for any "
        "phone in the set, although it will warn if you share roots between "
        "phones with different numbers of states, which is a weird thing to "
        "do but should still work.  If any phone is present "
        "in \"phone_sets\" but \"phone2num_pdf_classes\" does not map it to a length, "
        "it is an error.  Note that the behaviour of the resulting map is "
        "undefined for phones not present in \"phone_sets\". "
        "At entry, this function should be called with (*num_leaves == 0). "
        "It will number the leaves starting from (*num_leaves).",
        py::arg("P"),
        py::arg("phone_sets"),
        py::arg("phone2num_pdf_classes"),
        py::arg("share_roots"),
        py::arg("num_leaves"));
}

void pybind_build_tree(py::module& m) {

  m.def("BuildTree",
        &BuildTree,
        "BuildTree is the normal way to build a set of decision trees. "
        "The sets \"phone_sets\" dictate how we set up the roots of the decision trees. "
        "each set of phones phone_sets[i] has shared decision-tree roots, and if "
        "the corresponding variable share_roots[i] is true, the root will be shared "
        "for the different HMM-positions in the phone.  All phones in \"phone_sets\" "
        "should be in the stats (use FixUnseenPhones to ensure this). "
        "if for any i, do_split[i] is false, we will not do any tree splitting for "
        "phones in that set. "
        " @param qopts [in] Questions options class, contains questions for each key "
        "                 (e.g. each phone position) "
        " @param phone_sets [in] Each element of phone_sets is a set of phones whose "
        "               roots are shared together (prior to decision-tree splitting). "
        " @param phone2num_pdf_classes [in] A map from phones to the number of "
        "               \\ref pdf_class \"pdf-classes\" "
        "               in the phone (this info is derived from the HmmTopology object) "
        " @param share_roots [in] A vector the same size as phone_sets; says for each "
        "              phone set whether the root should be shared among all the "
        "              pdf-classes or not. "
        " @param do_split [in] A vector the same size as phone_sets; says for each "
        "              phone set whether decision-tree splitting should be done "
        "               (generally true for non-silence phones). "
        " @param stats [in] The statistics used in tree-building. "
        " @param thresh [in] Threshold used in decision-tree splitting (e.g. 1000), "
        "                 or you may use 0 in which case max_leaves becomes the "
        "                  constraint. "
        " @param max_leaves [in] Maximum number of leaves it will create; set this "
        "                to a large number if you want to just specify  \"thresh\". "
        " @param cluster_thresh [in] Threshold for clustering leaves after decision-tree "
        "                splitting (only within each phone-set); leaves will be combined "
        "                if log-likelihood change is less than this.  A value about equal "
        "                to \"thresh\" is suitable "
        "                if thresh != 0; otherwise, zero will mean no clustering is done, "
        "                or a negative value (e.g. -1) sets it to the smallest likelihood "
        "                change seen during the splitting algorithm; this typically causes "
        "                about a 20% reduction in the number of leaves. "

        " @param P [in] The central position of the phone context window, e.g. 1 for a "
        "              triphone system. "
        " @param round_num_leaves [in]  If true, then the number of leaves in the  "
        "                final tree is made a multiple of 8. This is done by  "
        "                further clustering the leaves after they are first "
        "                clustered based on log-likelihood change. "
        "                (See cluster_thresh above) (default: true) "
        "@return  Returns a pointer to an EventMap object that is the tree.",
        py::arg("qopts"),
        py::arg("phone_sets"),
        py::arg("phone2num_pdf_classes"),
        py::arg("share_roots"),
        py::arg("do_split"),
        py::arg("stats"),
        py::arg("thresh"),
        py::arg("max_leaves"),
        py::arg("cluster_thresh"),
        py::arg("P"),
        py::arg("round_num_leaves"));

  m.def("BuildTreeTwoLevel",
        &BuildTreeTwoLevel,
        "BuildTreeTwoLevel builds a two-level tree, useful for example in building tied mixture "
        "systems with multiple codebooks.  It first builds a small tree by splitting to "
        "\"max_leaves_first\".  It then splits at the leaves of \"max_leaves_first\" (think of this "
        "as creating multiple little trees at the leaves of the first tree), until the total "
        "number of leaves reaches \"max_leaves_second\".  It then outputs the second tree, along "
        "with a mapping from the leaf-ids of the second tree to the leaf-ids of the first tree. "
        "Note that the interface is similar to BuildTree, and in fact it calls BuildTree "
        "internally. "
        "\n"
        "The sets \"phone_sets\" dictate how we set up the roots of the decision trees. "
        "each set of phones phone_sets[i] has shared decision-tree roots, and if "
        "the corresponding variable share_roots[i] is true, the root will be shared "
        "for the different HMM-positions in the phone.  All phones in \"phone_sets\" "
        "should be in the stats (use FixUnseenPhones to ensure this). "
        "if for any i, do_split[i] is false, we will not do any tree splitting for "
        "phones in that set. "
        "\n"
        "@param qopts [in] Questions options class, contains questions for each key "
        "                 (e.g. each phone position) "
        "@param phone_sets [in] Each element of phone_sets is a set of phones whose "
        "               roots are shared together (prior to decision-tree splitting). "
        "@param phone2num_pdf_classes [in] A map from phones to the number of "
        "               \\ref pdf_class \"pdf-classes\" "
        "               in the phone (this info is derived from the HmmTopology object) "
        "@param share_roots [in] A vector the same size as phone_sets; says for each "
        "              phone set whether the root should be shared among all the "
        "              pdf-classes or not. "
        "@param do_split [in] A vector the same size as phone_sets; says for each "
        "              phone set whether decision-tree splitting should be done "
        "               (generally true for non-silence phones). "
        "@param stats [in] The statistics used in tree-building. "
        "@param max_leaves_first [in] Maximum number of leaves it will create in first "
        "                level of decision tree.  "
        "@param max_leaves_second [in] Maximum number of leaves it will create in second "
        "                level of decision tree.  Must be > max_leaves_first. "
        "@param cluster_leaves [in] Boolean value; if true, we post-cluster the leaves produced "
        "                in the second level of decision-tree split; if false, we don't. "
        "                The threshold for post-clustering is the log-like change of the last "
        "                decision-tree split; this typically causes about a 20% reduction in "
        "                the number of leaves. "
        "@param P [in]   The central position of the phone context window, e.g. 1 for a "
        "               triphone system. "
        "@param leaf_map [out]  Will be set to be a mapping from the leaves of the "
        "               \"big\" tree to the leaves of the \"little\" tree, which you can "
        "               view as cluster centers. "
        "@return  Returns a pointer to an EventMap object that is the (big) tree.",
        py::arg("qopts"),
        py::arg("phone_sets"),
        py::arg("phone2num_pdf_classes"),
        py::arg("share_roots"),
        py::arg("do_split"),
        py::arg("stats"),
        py::arg("max_leaves_first"),
        py::arg("max_leaves_second"),
        py::arg("cluster_leaves"),
        py::arg("P"),
        py::arg("leaf_map"));

  m.def("GenRandStats",
        &GenRandStats,
        "GenRandStats generates random statistics of the form used by BuildTree. "
        "It tries to do so in such a way that they mimic \"real\" stats.  The event keys "
        "and their corresponding values are: "
        "- key == -1 == kPdfClass -> pdf-class, generally corresponds to "
        "      zero-based position in HMM (0, 1, 2 .. hmm_lengths[phone]-1) "
        "- key == 0 -> phone-id of left-most context phone. "
        "- key == 1 -> phone-id of one-from-left-most context phone. "
        "- key == P-1 -> phone-id of central phone. "
        "- key == N-1 -> phone-id of right-most context phone. "
        "GenRandStats is useful only for testing but it serves to document the format of "
        "stats used by BuildTreeDefault. "
        "if is_ctx_dep[phone] is set to false, GenRandStats will not define the keys for "
        "other than the P-1'th phone. "
        "\n"
        "@param dim [in] dimension of features. "
        "@param num_stats [in] approximate number of separate phones-in-context wanted. "
        "@param N [in] context-size (typically 3) "
        "@param P [in] central-phone position in zero-based numbering (typically 1) "
        "@param phone_ids [in] integer ids of phones "
        "@param hmm_lengths [in] lengths of hmm for phone, indexed by phone. "
        "@param is_ctx_dep [in] boolean array indexed by phone, saying whether each phone "
        "    is context dependent. "
        "@param ensure_all_phones_covered [in] Boolean argument: if true, GenRandStats "
        "    ensures that every phone is seen at least once in the central position (P). "
        "@param stats_out [out] The statistics that this routine outputs.",
        py::arg("dim"),
        py::arg("num_stats"),
        py::arg("N"),
        py::arg("P"),
        py::arg("phone_ids"),
        py::arg("hmm_lengths"),
        py::arg("is_ctx_dep"),
        py::arg("ensure_all_phones_covered"),
        py::arg("stats_out"));

  m.def("ReadSymbolTableAsIntegers",
        &ReadSymbolTableAsIntegers,
        "included here because it's used in some tree-building "
        "calling code.  Reads an OpenFst symbl table, "
        "discards the symbols and outputs the integers",
        py::arg("filename"),
        py::arg("include_eps"),
        py::arg("syms"));

  m.def("AutomaticallyObtainQuestions",
        &AutomaticallyObtainQuestions,
        "Outputs sets of phones that are reasonable for questions "
        "to ask in the tree-building algorithm.  These are obtained by tree "
        "clustering of the phones; for each node in the tree, all the leaves "
        "accessible from that node form one of the sets of phones. "
        "  @param stats [in] The statistics as used for normal tree-building. "
        "  @param phone_sets_in [in] All the phones, pre-partitioned into sets. "
        "     The output sets will be various unions of these sets.  These sets "
        "     will normally correspond to \"real phones\", in cases where the phones "
        "     have stress and position markings. "
        "  @param all_pdf_classes_in [in] All the \\ref pdf_class \"pdf-classes\" "
        "    that we consider for clustering.  In the normal case this is the singleton "
        "     set {1}, which means that we only consider the central hmm-position "
        "     of the standard 3-state HMM, for clustering purposes. "
        "  @param P [in] The central position in the phone context window; normally "
        "     1 for triphone system.s "
        "  @param questions_out [out] The questions (sets of phones) are output to here.",
        py::arg("stats"),
        py::arg("phone_sets_in"),
        py::arg("all_pdf_classes_in"),
        py::arg("P"),
        py::arg("questions_out"));

  m.def("automatically_obtain_questions",
        [](BuildTreeStatsType &stats,
                                  const std::vector<std::vector<int32> > &phone_sets_in,
                                  const std::vector<int32> &all_pdf_classes_in,
                                  int32 P){
          py::gil_scoped_release gil_release;
    std::vector<std::vector<int32> > phone_sets_out;
      AutomaticallyObtainQuestions(stats,
                                   phone_sets_in,
                                   all_pdf_classes_in,
                                   P,
                                   &phone_sets_out);
            return phone_sets_out;
        },
        "Outputs sets of phones that are reasonable for questions "
        "to ask in the tree-building algorithm.  These are obtained by tree "
        "clustering of the phones; for each node in the tree, all the leaves "
        "accessible from that node form one of the sets of phones. "
        "  @param stats [in] The statistics as used for normal tree-building. "
        "  @param phone_sets_in [in] All the phones, pre-partitioned into sets. "
        "     The output sets will be various unions of these sets.  These sets "
        "     will normally correspond to \"real phones\", in cases where the phones "
        "     have stress and position markings. "
        "  @param all_pdf_classes_in [in] All the \\ref pdf_class \"pdf-classes\" "
        "    that we consider for clustering.  In the normal case this is the singleton "
        "     set {1}, which means that we only consider the central hmm-position "
        "     of the standard 3-state HMM, for clustering purposes. "
        "  @param P [in] The central position in the phone context window; normally "
        "     1 for triphone system.s "
        "  @param questions_out [out] The questions (sets of phones) are output to here.",
        py::arg("stats"),
        py::arg("phone_sets_in"),
        py::arg("all_pdf_classes_in"),
        py::arg("P"));

  m.def("KMeansClusterPhones",
        &KMeansClusterPhones,
        "This function clusters the phones (or some initially specified sets of phones) "
        "into sets of phones, using a k-means algorithm.  Useful, for example, in building "
        "simple models for purposes of adaptation.",
        py::arg("stats"),
        py::arg("phone_sets_in"),
        py::arg("all_pdf_classes_in"),
        py::arg("P"),
        py::arg("num_classes"),
        py::arg("sets_out"));

  m.def("ReadRootsFile",
        &ReadRootsFile,
        "Reads the roots file (throws on error).  Format is lines like: "
        " \"shared split 1 2 3 4\", "
        " \"not-shared not-split 5\", "
        "and so on.  The numbers are indexes of phones.",
        py::arg("is"),
        py::arg("phone_sets"),
        py::arg("is_shared_root"),
        py::arg("is_split_root"));
}


void pybind_cluster_utils(py::module& m) {

  m.def("SumClusterableObjf",
        &SumClusterableObjf,
        "Returns the total objective function after adding up all the "
        "statistics in the vector (pointers may be NULL).",
        py::arg("vec"));
  m.def("SumClusterableNormalizer",
        &SumClusterableNormalizer,
        "Returns the total normalizer (usually count) of the cluster (pointers may be NULL).",
        py::arg("vec"));
  m.def("SumClusterable",
        &SumClusterable,
        "Sums stats (ptrs may be NULL). Returns NULL if no non-NULL stats present.",
        py::arg("vec"));
  m.def("EnsureClusterableVectorNotNull",
        &EnsureClusterableVectorNotNull,
        "Fills in any (NULL) holes in \"stats\" vector, with empty stats, because "
        "certain algorithms require non-NULL stats.  If \"stats\" nonempty, requires it "
        "to contain at least one non-NULL pointer that we can call Copy() on.",
        py::arg("stats"));
  m.def("AddToClusters",
        &AddToClusters,
        "Given stats and a vector \"assignments\" of the same size (that maps to "
        "cluster indices), sums the stats up into \"clusters.\"  It will add to any "
        "stats already present in \"clusters\" (although typically \"clusters\" will be "
        "empty when called), and it will extend with NULL pointers for any unseen "
        "indices. Call EnsureClusterableStatsNotNull afterwards if you want to ensure "
        "all non-NULL clusters. Pointer in \"clusters\" are owned by caller. Pointers in "
        "\"stats\" do not have to be non-NULL.",
        py::arg("stats"),
        py::arg("assignments"),
        py::arg("clusters"));
  m.def("AddToClustersOptimized",
        &AddToClustersOptimized,
        "AddToClustersOptimized does the same as AddToClusters (it sums up the stats "
        "within each cluster, except it uses the sum of all the stats (\"total\") to "
        "optimize the computation for speed, if possible.  This will generally only be "
        "a significant speedup in the case where there are just two clusters, which "
        "can happen in algorithms that are doing binary splits; the idea is that we "
        "sum up all the stats in one cluster (the one with the fewest points in it), "
        "and then subtract from the total.",
        py::arg("stats"),
        py::arg("assignments"),
        py::arg("total"),
        py::arg("clusters"));
  m.def("ClusterBottomUp",
        &ClusterBottomUp,
        "A bottom-up clustering algorithm. There are two parameters that control how "
        "many clusters we get: a \"max_merge_thresh\" which is a threshold for merging "
        "clusters, and a min_clust which puts a floor on the number of clusters we want. Set "
        "max_merge_thresh = large to use the min_clust only, or min_clust to 0 to use "
        "the max_merge_thresh only."
        "\n"
        "The algorithm is: "
        "//code "
        "    while (num-clusters > min_clust && smallest_merge_cost <= max_merge_thresh) "
        "         merge the closest two clusters. "
        " //endcode"
        "\n"
        "@param points [in] Points to be clustered (may not contain NULL pointers) "
        "@param thresh [in] Threshold on cost change from merging clusters; clusters "
        "             won't be merged if the cost is more than this "
        "@param min_clust [in] Minimum number of clusters desired; we'll stop merging "
        "                after reaching this number. "
        "@param clusters_out [out] If non-NULL, will be set to a vector of size equal "
        "               to the number of output clusters, containing the clustered "
        "               statistics.  Must be empty when called. "
        "@param assignments_out [out] If non-NULL, will be resized to the number of "
        "               points, and each element is the index of the cluster that point "
        "               was assigned to. "
        "@return Returns the total objf change relative to all clusters being separate, which is "
        "  a negative.  Note that this is not the same as what the other clustering algorithms return.",
        py::arg("points"),
        py::arg("thresh"),
        py::arg("min_clust"),
        py::arg("clusters_out"),
        py::arg("assignments_out"));
  m.def("ClusterBottomUpCompartmentalized",
        &ClusterBottomUpCompartmentalized,
        "This is a bottom-up clustering where the points are pre-clustered in a set "
        "of compartments, such that only points in the same compartment are clustered "
        "together. The compartment and pair of points with the smallest merge cost "
        "is selected and the points are clustered. The result stays in the same "
        "compartment. The code does not merge compartments, and hence assumes that "
        "the number of compartments is smaller than the 'min_clust' option. "
        "The clusters in \"clusters_out\" are newly allocated and owned by the caller.",
        py::arg("points"),
        py::arg("thresh"),
        py::arg("min_clust"),
        py::arg("clusters_out"),
        py::arg("assignments_out"));
    {

      using PyClass = RefineClustersOptions;
      auto refine_clusters_options = py::class_<PyClass>(
          m, "RefineClustersOptions");
      refine_clusters_options.def(py::init<>())
        .def(py::init<int32, int32>(), py::arg("num_iters_in"),
           py::arg("top_n_in"))
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("top_n", &PyClass::top_n)
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Write", &PyClass::Read, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
    }
  m.def("RefineClusters",
        &RefineClusters,
        "RefineClusters is mainly used internally by other clustering algorithms. "
        "\n"
        "It starts with a given assignment of points to clusters and "
        "keeps trying to improve it by moving points from cluster to cluster, up to "
        "a maximum number of iterations."
        "\n"
        "\"clusters\" and \"assignments\" are both input and output variables, and so "
        "both MUST be non-NULL."
        "\n"
        "\"top_n\" (>=2) is a pruning value: more is more exact, fewer is faster. The "
        "algorithm initially finds the \"top_n\" closest clusters to any given point, "
        "and from that point only consider move to those \"top_n\" clusters. Since "
        "RefineClusters is called multiple times from ClusterKMeans (for instance), "
        "this is not really a limitation.",
        py::arg("points"),
        py::arg("clusters"),
        py::arg("assignments"),
        py::arg("cfg"));
    {

      using PyClass = ClusterKMeansOptions;
      auto cluster_kmeans_options = py::class_<PyClass>(
          m, "ClusterKMeansOptions");
      cluster_kmeans_options.def(py::init<>())
      .def_readwrite("refine_cfg", &PyClass::refine_cfg)
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("num_tries", &PyClass::num_tries)
      .def_readwrite("verbose", &PyClass::verbose);
    }
  m.def("ClusterKMeans",
        &ClusterKMeans,
        "ClusterKMeans is a K-means-like clustering algorithm. It starts with "
        "pseudo-random initialization of points to clusters and uses RefineClusters "
        "to iteratively improve the cluster assignments.  It does this for "
        "multiple iterations and picks the result with the best objective function. "
        "\n"
        "\n"
        "ClusterKMeans implicitly uses Rand(). It will not necessarily return "
        "the same value on different calls.  Use sRand() if you want consistent "
        "results."
        "The algorithm used in ClusterKMeans is a \"k-means-like\" algorithm that tries "
        "to be as efficient as possible.  Firstly, since the algorithm it uses "
        "includes random initialization, it tries the whole thing cfg.num_tries times "
        "and picks the one with the best objective function.  Each try, it does as "
        "follows: it randomly initializes points to clusters, and then for "
        "cfg.num_iters iterations it calls RefineClusters().  The options to "
        "RefineClusters() are given by cfg.refine_cfg.  Calling RefineClusters once "
        "will always be at least as good as doing one iteration of reassigning points to "
        "clusters, but will generally be quite a bit better (without taking too "
        "much extra time)."
        "\n"
        "@param points [in]  points to be clustered (must be all non-NULL). "
        "@param num_clust [in] number of clusters requested (it will always return exactly "
        "               this many, or will fail if num_clust > points.size()). "
        "@param clusters_out [out] may be NULL; if non-NULL, should be empty when called. "
        "        Will be set to a vector of statistics corresponding to the output clusters. "
        "@param assignments_out [out] may be NULL; if non-NULL, will be set to a vector of "
        "           same size as \"points\", which says for each point which cluster "
        "            it is assigned to. "
        "@param cfg [in] configuration class specifying options to the algorithm. "
        "@return Returns the objective function improvement versus everything being "
        "   in the same cluster.",
        py::arg("points"),
        py::arg("num_clust"),
        py::arg("clusters_out"),
        py::arg("assignments_out"),
        py::arg("cfg"));
    {

      using PyClass = TreeClusterOptions;
      auto tree_cluster_options = py::class_<PyClass>(
          m, "TreeClusterOptions");
      tree_cluster_options.def(py::init<>())
      .def_readwrite("kmeans_cfg", &PyClass::kmeans_cfg)
      .def_readwrite("branch_factor", &PyClass::branch_factor)
      .def_readwrite("thresh", &PyClass::thresh);
    }
  m.def("TreeCluster",
        &TreeCluster,
        "TreeCluster is a top-down clustering algorithm, using a binary tree (not "
        "necessarily balanced). Returns objf improvement versus having all points "
        "in one cluster.  The algorithm is: "
        "   - Initialize to 1 cluster (tree with 1 node). "
        "   - Maintain, for each cluster, a \"best-binary-split\" (using ClusterKMeans "
        "     to do so). Always split the highest scoring cluster, until we can do no "
        "     more splits. "
        "\n"
        "@param points [in] Data points to be clustered "
        "@param max_clust  [in] Maximum number of clusters (you will get exactly this number, "
        "              if there are at least this many points, except if you set the "
        "              cfg.thresh value nonzero, in which case that threshold may limit "
        "              the number of clusters. "
        "@param clusters_out [out] If non-NULL, will be set to the a vector whose first "
        "              (*num_leaves_out) elements are the leaf clusters, and whose "
        "              subsequent elements are the nonleaf nodes in the tree, in "
        "              topological order with the root node last.  Must be empty vector "
        "              when this function is called. "
        "@param assignments_out [out] If non-NULL, will be set to a vector to a vector the "
        "             same size as \"points\", where assignments[i] is the leaf node index i "
        "             to which the i'th point gets clustered. "
        "@param clust_assignments_out [out] If non-NULL, will be set to a vector the same size "
        "              as clusters_out  which says for each node (leaf or nonleaf), the "
        "              index of its parent.  For the root node (which is last), "
        "              assignments_out[i] == i.  For each i, assignments_out[i]>=i, i.e. "
        "              any node's parent is higher numbered than itself.  If you don't need "
        "              this information, consider using instead the ClusterTopDown function. "
        "@param num_leaves_out [out] If non-NULL, will be set to the number of leaf nodes "
        "              in the tree. "
        "@param cfg [in] Configuration object that controls clustering behavior.  Most "
        "               important value is \"thresh\", which provides an alternative mechanism "
        "               [other than max_clust] to limit the number of leaves.",
        py::arg("points"),
        py::arg("max_clust"),
        py::arg("clusters_out"),
        py::arg("assignments_out"),
        py::arg("clust_assignments_out"),
        py::arg("num_leaves_out"),
        py::arg("cfg"));
  m.def("ClusterTopDown",
        &ClusterTopDown,
        "A clustering algorithm that internally uses TreeCluster, "
        "but does not give you the information about the structure of the tree. "
        "The \"clusters_out\" and \"assignments_out\" may be NULL if the outputs are not "
        "needed. "
        "\n"
        "@param points [in]  points to be clustered (must be all non-NULL). "
        "@param max_clust [in] Maximum number of clusters (you will get exactly this number, "
        "              if there are at least this many points, except if you set the "
        "              cfg.thresh value nonzero, in which case that threshold may limit "
        "              the number of clusters. "
        "@param clusters_out [out] may be NULL; if non-NULL, should be empty when called. "
        "         Will be set to a vector of statistics corresponding to the output clusters. "
        "@param assignments_out [out] may be NULL; if non-NULL, will be set to a vector of "
        "         same size as \"points\", which says for each point which cluster "
        "          it is assigned to. "
        "@param cfg [in] Configuration object that controls clustering behavior.  Most "
        "              important value is \"thresh\", which provides an alternative mechanism "
        "              [other than max_clust] to limit the number of leaves.",
        py::arg("points"),
        py::arg("max_clust"),
        py::arg("clusters_out"),
        py::arg("assignments_out"),
        py::arg("cfg"));
}


void pybind_clusterable_classes(py::module& m) {

    auto scalar_clusterable = py::class_<ScalarClusterable, Clusterable>(
        m, "ScalarClusterable");

    auto gauss_clusterable = py::class_<GaussClusterable, Clusterable>(
        m, "GaussClusterable")
      .def(py::pickle(
        [](const GaussClusterable &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            GaussClusterable p;

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               Clusterable *gc = p.ReadNew(str, true);

            return reinterpret_cast<GaussClusterable*>(gc);
        }
    ));

    auto vector_clusterable = py::class_<VectorClusterable, Clusterable>(
        m, "VectorClusterable");
}


void pybind_context_dep(py::module& m) {
  using PyClass = ContextDependency;
    auto context_dependency = py::class_<PyClass, ContextDependencyInterface>(
        m, "ContextDependency",
        "ContextDependency is quite a generic decision tree."
        "\n"
        "It does not actually do very much-- all the magic is in the EventMap object. "
        "All this class does is to encode the phone context as a sequence of events, and "
        "pass this to the EventMap object to turn into what it will interpret as a "
        "vector of pdfs."
        "\n"
        "Different versions of the ContextDependency class that are written in the future may "
        "have slightly different interfaces and pass more stuff in as events, to the "
        "EventMap object."
        "\n"
        "In order to separate the process of training decision trees from the process "
        "of actually using them, we do not put any training code into the ContextDependency class.");

    context_dependency.def(py::init<>())
      .def(py::init<int32, int32, EventMap *>(), py::arg("N"),
           py::arg("P"), py::arg("to_pdf"))
           .def(py::init([](std::string file_path){
                  ContextDependency ctx_dep;
                  ReadKaldiObject(file_path, &ctx_dep);
                  return &ctx_dep;}),
                  py::arg("file_path"),
                  py::return_value_policy::reference)
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("ContextWidth", &PyClass::ContextWidth)
        .def("CentralPosition", &PyClass::CentralPosition)
        .def("NumPdfs", &PyClass::NumPdfs)
        .def("ToPdfMap", &PyClass::ToPdfMap)
        //.def("GetPdfInfo", &PyClass::GetPdfInfo, py::arg("phones"), py::arg("num_pdf_classes"), py::arg("pdf_info"))
        .def("Compute", &PyClass::Compute, py::arg("phoneseq"), py::arg("pdf_class"), py::arg("pdf_id"))
        .def("__str__", [](const PyClass& cd) {
          std::ostringstream os;
          bool binary = false;
          cd.Write(os, binary);
          return os.str();
        })
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass *p = new PyClass();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true);

            return p;
        }
    ));

  m.def(
      "GenRandContextDependency",
      &GenRandContextDependency,
      "GenRandContextDependency is mainly of use for debugging.  Phones must be sorted and uniq "
      "on input. "
      "@param phones [in] A vector of phone id's [must be sorted and uniq]. "
      "@param ensure_all_covered [in] boolean argument; if true,  GenRandContextDependency "
      "       generates a context-dependency object that \"works\" for all phones [no gaps]. "
      "@param num_pdf_classes [out] outputs a vector indexed by phone, of the number "
      "         of pdf classes (e.g. states) for that phone. "
      "@return Returns the a context dependency object.",
      py::arg("phones"),
      py::arg("ensure_all_covered"),
      py::arg("num_pdf_classes"));

  m.def(
      "GenRandContextDependencyLarge",
      &GenRandContextDependencyLarge,
      "GenRandContextDependencyLarge is like GenRandContextDependency but generates a larger tree "
      "with specified N and P for use in \"one-time\" larger-scale tests",
      py::arg("phones"),
      py::arg("N"),
      py::arg("P"),
      py::arg("ensure_all_covered"),
      py::arg("num_pdf_classes"));

  m.def(
      "MonophoneContextDependency",
      &MonophoneContextDependency,
      "MonophoneContextDependency() returns a new ContextDependency object that "
      "corresponds to a monophone system. "
      "The map phone2num_pdf_classes maps from the phone id to the number of "
      "pdf-classes we have for that phone (e.g. 3, so the pdf-classes would be "
      "0, 1, 2).",
      py::arg("phones"),
      py::arg("phone2num_pdf_classes"));

  m.def(
      "MonophoneContextDependencyShared",
      [](
         const std::vector<std::vector<int32> > &phone_sets,
         const std::vector<int32> &phone2num_pdf_classes
      ) -> ContextDependency* {

    ContextDependency *ctx_dep = MonophoneContextDependencyShared(phone_sets, phone2num_pdf_classes);

    return ctx_dep;
      },
      "MonophoneContextDependencyShared is as MonophoneContextDependency but lets "
 "you define classes of phones which share pdfs (e.g. different stress-markers of a single "
 "phone.)  Each element of phone_classes is a set of phones that are in that class.",
      py::arg("phone_classes"),
      py::arg("phone2num_pdf_classes"), py::return_value_policy::take_ownership);
}

void init_tree(py::module &_m) {
  py::module m = _m.def_submodule("tree", "tree pybind for Kaldi");

  pybind_event_map(m);
  pybind_build_tree_questions(m);
  pybind_build_tree_utils(m);
  pybind_build_tree(m);
  pybind_cluster_utils(m);
  pybind_clusterable_classes(m);
  pybind_context_dep(m);


    m.def("build_tree",
          [](
          const HmmTopology &topo,
                      std::vector<std::vector<int32> > questions,
                  BuildTreeStatsType stats,
                      std::string roots_filename,
                      std::string tree_out_filename,
                      int32 num_iters_refine = 0,
                      int32 P = 1,
                      int32 N = 3,

                  BaseFloat thresh = 300.0,
                  BaseFloat cluster_thresh = -1.0,
                  int32 max_leaves = 0,
                  bool round_num_leaves = true
          ){
            bool binary = true;
            std::vector<std::vector<int32> > phone_sets;
            std::vector<bool> is_shared_root;
            std::vector<bool> is_split_root;
            {
                  Input ki(roots_filename.c_str());
                  ReadRootsFile(ki.Stream(), &phone_sets, &is_shared_root, &is_split_root);
            }
          for (size_t i = 0; i < questions.size(); i++) {
                  std::sort(questions[i].begin(), questions[i].end());
                  if (!IsSortedAndUniq(questions[i]))
                  KALDI_ERR << "Questions contain duplicate phones";
            }
            size_t nq = static_cast<int32>(questions.size());
            SortAndUniq(&questions);
            if (questions.size() != nq)
                  KALDI_WARN << (nq-questions.size())
                        << " duplicate questions present";

            // ProcessTopo checks that all phones in the topo are
            // represented in at least one questions (else warns), and
            // returns the max # pdf classes in any given phone (normally
            // 3).
            std::vector<int32> seen_phones;  // ids of phones seen in questions.
            for (size_t i = 0; i < questions.size(); i++)
            for (size_t j= 0; j < questions[i].size(); j++) seen_phones.push_back(questions[i][j]);
            SortAndUniq(&seen_phones);
            // topo_phones is also sorted and uniq; a list of phones defined in the topology.
            const std::vector<int32> &topo_phones = topo.GetPhones();
            if (seen_phones != topo_phones) {
            std::ostringstream ss_seen, ss_topo;
            WriteIntegerVector(ss_seen, false, seen_phones);
            WriteIntegerVector(ss_topo, false, topo_phones);
            KALDI_WARN << "ProcessTopo: phones seen in questions differ from those in topology: "
                        << ss_seen.str() << " vs. " << ss_topo.str();
            if (seen_phones.size() > topo_phones.size()) {
                  KALDI_ERR << "ProcessTopo: phones are asked about that are undefined in the topology.";
            } // we accept the reverse (not asking about all phones), even though it's very bad.
            }

            int32 max_num_pdf_classes = 0;
            for (size_t i = 0; i < topo_phones.size(); i++) {
            int32 p = topo_phones[i];
            int32 num_pdf_classes = topo.NumPdfClasses(p);
            max_num_pdf_classes = std::max(num_pdf_classes, max_num_pdf_classes);
            }
            KALDI_LOG << "Max # pdf classes is " << max_num_pdf_classes;

            Questions qo;

            QuestionsForKey phone_opts(num_iters_refine);
            // the questions-options corresponding to keys 0, 1, .. N-1 which
            // represent the phonetic context positions (including the central phone).
            for (int32 n = 0; n < N; n++) {
                  KALDI_LOG << "Setting questions for phonetic-context position "<< n;
                  phone_opts.initial_questions = questions;
                  qo.SetQuestionsOf(n, phone_opts);
            }

            QuestionsForKey pdfclass_opts(num_iters_refine);
            std::vector<std::vector<int32> > pdfclass_questions(max_num_pdf_classes-1);
            for (int32 i = 0; i < max_num_pdf_classes - 1; i++)
                  for (int32 j = 0; j <= i; j++)
                  pdfclass_questions[i].push_back(j);
            // E.g. if max_num_pdf_classes == 3,  pdfclass_questions is now [ [0], [0, 1] ].
            pdfclass_opts.initial_questions = pdfclass_questions;
            KALDI_LOG << "Setting questions for hmm-position [hmm-position ranges from 0 to "<< (max_num_pdf_classes-1) <<"]";
            qo.SetQuestionsOf(kPdfClass, pdfclass_opts);

            std::vector<int32> phone2num_pdf_classes;
            topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

            EventMap *to_pdf = NULL;

            //////// Build the tree. ////////////

            to_pdf = BuildTree(qo,
                              phone_sets,
                              phone2num_pdf_classes,
                              is_shared_root,
                              is_split_root,
                              stats,
                              thresh,
                              max_leaves,
                              cluster_thresh,
                              P,
                              round_num_leaves);

            { // This block is to warn about low counts.
                  std::vector<BuildTreeStatsType> split_stats;
                  SplitStatsByMap(stats, *to_pdf,
                              &split_stats);
                  for (size_t i = 0; i < split_stats.size(); i++)
                  if (SumNormalizer(split_stats[i]) < 100.0)
                  KALDI_VLOG(1) << "For pdf-id " << i << ", low count "
                                    << SumNormalizer(split_stats[i]);
            }

            ContextDependency ctx_dep(N, P, to_pdf);  // takes ownership
            // of pointer "to_pdf", so set it NULL.
            to_pdf = NULL;

            WriteKaldiObject(ctx_dep, tree_out_filename, binary);

            {  // This block is just doing some checks.

                  std::vector<int32> all_phones;
                  for (size_t i = 0; i < phone_sets.size(); i++)
                  all_phones.insert(all_phones.end(),
                                    phone_sets[i].begin(), phone_sets[i].end());
                  SortAndUniq(&all_phones);
                  if (all_phones != topo.GetPhones()) {
                  std::ostringstream ss;
                  WriteIntegerVector(ss, false, all_phones);
                  ss << " vs. ";
                  WriteIntegerVector(ss, false, topo.GetPhones());
                  KALDI_WARN << "Mismatch between phone sets provided in roots file, and those in topology: " << ss.str();
                  }
                  std::vector<int32> phones_vec;  // phones we saw.
                  PossibleValues(P, stats, &phones_vec); // function in build-tree-utils.h

                  std::vector<int32> unseen_phones;  // diagnostic.
                  for (size_t i = 0; i < all_phones.size(); i++)
                  if (!std::binary_search(phones_vec.begin(), phones_vec.end(), all_phones[i]))
                  unseen_phones.push_back(all_phones[i]);
                  for (size_t i = 0; i < phones_vec.size(); i++)
                  if (!std::binary_search(all_phones.begin(), all_phones.end(), phones_vec[i]))
                  KALDI_ERR << "Phone " << (phones_vec[i])
                              << " appears in stats but is not listed in roots file.";
                  if (!unseen_phones.empty()) {
                  std::ostringstream ss;
                  for (size_t i = 0; i < unseen_phones.size(); i++)
                  ss << unseen_phones[i] << ' ';
                  // Note, unseen phones is just a warning as in certain kinds of
                  // systems, this can be OK (e.g. where phone encodes position and
                  // stress information).
                  KALDI_WARN << "Saw no stats for following phones: " << ss.str();
                  }
            }
          },
               py::arg("topo"),
               py::arg("questions"),
               py::arg("stats"),
               py::arg("roots_filename"),
               py::arg("tree_out_filename"),
               py::arg("num_iters_refine") = 0,
               py::arg("P") = 1,
               py::arg("N") = 3,
               py::arg("thresh") = 300.0,
               py::arg("cluster_thresh") = -1.0,
               py::arg("max_leaves") = 0,
               py::arg("round_num_leaves") = true);
}
