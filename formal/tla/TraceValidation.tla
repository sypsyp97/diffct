-------------------------- MODULE TraceValidation --------------------------
EXTENDS ObservedProtocol
VARIABLE pos
vars == <<pos, protocolVars>>
Init == ProtocolInit /\ pos = 0
Next == (pos < Len(Events) /\ ProtocolStep(Events[pos + 1]) /\ pos' = pos + 1)
        \/ (pos = Len(Events) /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
TraceConsumed == <>(pos = Len(Events))
CollectivesFinished == (pos = Len(Events) /\ ~AllowIncomplete) =>
    \A g \in Groups : (\A r, s \in Members[g] : Len(issued[g][r]) = Len(issued[g][s]))
                      /\ (\A r \in Members[g] : completed[g][r] = Len(issued[g][r]))
=============================================================================
