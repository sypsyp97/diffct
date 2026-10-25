--------------------------- MODULE RecomputeTrace ---------------------------
EXTENDS RecomputeOrder
CONSTANTS Actual, Project, Replicated, ReduceInput
TraceInit == Init /\ project = Project /\ replicated = Replicated /\ reduceInput = ReduceInput
TraceStep(r) == /\ index[r] < Len(Actual)
                /\ Step(r)
                /\ history'[r] = Append(history[r], Actual[index[r] + 1])
TraceNext == (\E r \in Ranks : TraceStep(r)) \/ (Done /\ UNCHANGED vars)
TraceSpec == TraceInit /\ [][TraceNext]_vars /\ WF_vars(TraceNext)
ObservedLength == \A r \in Ranks : index[r] <= Len(Actual) /\ Len(Actual) = Count
=============================================================================
