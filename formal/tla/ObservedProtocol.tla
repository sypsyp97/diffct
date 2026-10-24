-------------------------- MODULE ObservedProtocol --------------------------
EXTENDS Integers, Sequences, FiniteSets
CONSTANTS Ranks, Groups, Members, Events, AllowIncomplete
VARIABLES issued, completed, created, plans, success, errors, bad
protocolVars == <<issued, completed, created, plans, success, errors, bad>>
ProtocolInit == /\ issued = [g \in Groups |-> [r \in Ranks |-> <<>>]]
                /\ completed = [g \in Groups |-> [r \in Ranks |-> 0]]
                /\ created = [g \in Groups |-> [r \in Ranks |-> 0]]
                /\ plans = [r \in Ranks |-> <<>>] /\ success = {} /\ errors = {} /\ bad = {}
ProtocolStep(e) ==
  /\ issued' = IF e.kind = "collective_enter"
                  THEN [issued EXCEPT ![e.group][e.rank] = Append(@, e.signature)] ELSE issued
  /\ completed' = IF e.kind = "collective_exit"
                     THEN [completed EXCEPT ![e.group][e.rank] = @ + 1] ELSE completed
  /\ created' = IF e.kind = "group_create"
                   THEN [created EXCEPT ![e.group][e.rank] = @ + 1] ELSE created
  /\ plans' = IF e.kind = "plan_agreement"
                 THEN [plans EXCEPT ![e.rank] = Append(@, e.signature)] ELSE plans
  /\ success' = IF e.kind = "call_end" THEN success \cup {<<e.name, e.rank>>} ELSE success
  /\ errors' = IF e.kind \in {"call_error", "runtime_timeout"}
                  THEN errors \cup {<<e.name, e.rank>>} ELSE errors
  /\ bad' = bad \cup
       (IF e.kind = "runtime_timeout" THEN {"timeout"} ELSE {}) \cup
       (IF e.kind = "call_error" /\ ~e.allowed THEN {"crash"} ELSE {}) \cup
       (IF e.kind = "route" /\ e.required /\ ~e.bounded THEN {"route"} ELSE {}) \cup
       (IF e.kind = "route" /\ e.pure /\ e.bounded THEN {"pure"} ELSE {}) \cup
       (IF e.kind = "route" /\ e.require_rejection /\ ~e.rejected THEN {"rejection"} ELSE {}) \cup
       (IF e.kind \in {"host_read", "host_reuse"} /\ ~e.ready THEN {"read"} ELSE {}) \cup
       (IF e.kind = "pool_close" /\ e.remaining # 0 THEN {"leak"} ELSE {}) \cup
       (IF e.kind = "budget_observation" /\ e.admitted > e.bound THEN {"budget"} ELSE {})
Prefix(a, b) == Len(a) <= Len(b) /\ \A i \in 1..Len(a) : a[i] = b[i]
CollectiveAgreement == \A g \in Groups : \A r, s \in Members[g] :
                        Prefix(issued[g][r], issued[g][s]) \/ Prefix(issued[g][s], issued[g][r])
PlanAgreement == \A r, s \in Ranks : Prefix(plans[r], plans[s]) \/ Prefix(plans[s], plans[r])
NoMixedOutcomes == \A s \in success : ~ (\E e \in errors : s[1] = e[1])
NoRuntimeTimeout == "timeout" \notin bad
NoUnexpectedCrash == "crash" \notin bad
BoundedRoute == "route" \notin bad
PureForwardRoute == "pure" \notin bad
GeometryRejected == "rejection" \notin bad
NoUnfinishedRead == "read" \notin bad
NoLeaks == "leak" \notin bad
BudgetBound == "budget" \notin bad
GroupReuse == \A g \in Groups : \A r \in Ranks : created[g][r] <= 1
CompletionCounts == \A g \in Groups : \A r \in Ranks : completed[g][r] <= Len(issued[g][r])
=============================================================================
