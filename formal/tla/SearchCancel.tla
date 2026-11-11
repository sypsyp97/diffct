----------------------------- MODULE SearchCancel -----------------------------
EXTENDS Naturals, FiniteSets, TLC
CONSTANTS Hosts, MaxTickets, MaxRequests, Mutation
VARIABLES model, ticket, started, pending, published, requests, nextId, slot,
          badCleanup, crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting

vars == <<model, ticket, started, pending, published, requests, nextId, slot,
          badCleanup, crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting>>
Ids == 1..MaxRequests
Awaiting == {"canceled", "success", "failure"}
Unused == [host |-> CHOOSE h \in Hosts : TRUE, ticket |-> 0,
           status |-> "unused", outcome |-> "none"]

Init == /\ model = "loading"
        /\ ticket = [h \in Hosts |-> 0]
        /\ started = [h \in Hosts |-> 0]
        /\ pending = [h \in Hosts |-> FALSE]
        /\ published = [h \in Hosts |-> 0]
        /\ requests = [i \in Ids |-> Unused]
        /\ nextId = 0 /\ slot = 0
        /\ badCleanup = FALSE /\ crossCanceled = FALSE /\ obsoleteHandled = FALSE
        /\ lateIgnored = FALSE /\ stoppedWaiting = FALSE

\* An input event invalidates its host's old ticket and clears old semantic results.
Input(h) == /\ ticket[h] < MaxTickets
            /\ ticket' = [ticket EXCEPT ![h] = @ + 1]
            /\ pending' = [pending EXCEPT ![h] = model # "off"]
            /\ published' = [published EXCEPT ![h] = 0]
            /\ UNCHANGED <<model, started, requests, nextId, slot, badCleanup,
                           crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting>>

Ready == /\ model = "loading" /\ model' = "ready"
         /\ UNCHANGED <<ticket, started, pending, published, requests, nextId, slot,
                        badCleanup, crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting>>

\* Ready listeners / debounce callbacks share one engine slot across the two hosts.
Start(h) ==
  /\ model = "ready" /\ ticket[h] > started[h] /\ nextId < MaxRequests
  /\ LET canceled == slot # 0 /\ requests[slot].status = "running"
         prior == IF canceled
                  THEN [requests EXCEPT ![slot].status = "canceled", ![slot].outcome = "canceled"]
                  ELSE requests
     IN /\ requests' = [prior EXCEPT ![nextId + 1] =
                          [host |-> h, ticket |-> ticket[h], status |-> "running", outcome |-> "none"]]
        /\ crossCanceled' = (crossCanceled \/ (canceled /\ requests[slot].host # h))
  /\ started' = [started EXCEPT ![h] = ticket[h]]
  /\ pending' = [pending EXCEPT ![h] = TRUE]
  /\ nextId' = nextId + 1 /\ slot' = nextId + 1
  /\ UNCHANGED <<model, ticket, published, badCleanup, obsoleteHandled, lateIgnored, stoppedWaiting>>

\* Delivery and the promise continuation are separate: an input can occur between them.
Deliver(i) == /\ requests[i].status = "running"
              /\ \E outcome \in {"success", "failure"} :
                    requests' = [requests EXCEPT ![i].status = outcome, ![i].outcome = outcome]
              /\ UNCHANGED <<model, ticket, started, pending, published, nextId, slot,
                             badCleanup, crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting>>

Handle(i) ==
  /\ requests[i].status \in Awaiting
  /\ LET h == requests[i].host
         current == requests[i].ticket = ticket[h]
         legacyReturn == Mutation = "legacyCancel" /\ requests[i].outcome = "canceled"
         clears == ~legacyReturn /\ (current \/ Mutation = "obsoleteCleanup")
         publishes == requests[i].outcome = "success" /\ (current \/ Mutation = "stalePublication")
     IN /\ pending' = IF clears THEN [pending EXCEPT ![h] = FALSE] ELSE pending
        /\ published' = IF publishes THEN [published EXCEPT ![h] = requests[i].ticket] ELSE published
        /\ badCleanup' = (badCleanup \/ (~current /\ clears /\ pending[h]))
        /\ obsoleteHandled' = (obsoleteHandled \/ ~current)
  /\ requests' = [requests EXCEPT ![i].status = "handled"]
  /\ UNCHANGED <<model, ticket, started, nextId, slot, crossCanceled, lateIgnored, stoppedWaiting>>

\* engine.pending.id rejects old worker replies after global replacement.
IgnoreLate(i) == /\ requests[i].outcome = "canceled" /\ i # slot /\ ~lateIgnored
                 /\ lateIgnored' = TRUE
                 /\ UNCHANGED <<model, ticket, started, pending, published, requests, nextId, slot,
                                badCleanup, crossCanceled, obsoleteHandled, stoppedWaiting>>

Stop ==
  /\ model # "off"
  /\ model' = "off"
  /\ stoppedWaiting' = (stoppedWaiting \/ (\E h \in Hosts : pending[h]))
  /\ pending' = [h \in Hosts |-> FALSE]
  /\ requests' = IF slot # 0 /\ requests[slot].status = "running"
                  THEN [requests EXCEPT ![slot].status = "canceled", ![slot].outcome = "canceled"]
                  ELSE requests
  /\ slot' = 0
  /\ UNCHANGED <<ticket, started, published, nextId, badCleanup,
                 crossCanceled, obsoleteHandled, lateIgnored>>

Next == (\E h \in Hosts : Input(h) \/ Start(h)) \/ Ready \/ Stop
        \/ (\E i \in Ids : Deliver(i) \/ Handle(i) \/ IgnoreLate(i))
        \/ UNCHANGED vars

\* Fairness assumptions: load eventually answers if not stopped; admitted worker requests
\* eventually succeed/fail; queued callbacks and promise continuations eventually run.
Spec == Init /\ [][Next]_vars /\ WF_vars(Ready)
        /\ (\A h \in Hosts : WF_vars(Start(h)))
        /\ (\A i \in Ids : WF_vars(Deliver(i)) /\ WF_vars(Handle(i)))

TypeOK == /\ model \in {"loading", "ready", "off"}
          /\ ticket \in [Hosts -> 0..MaxTickets] /\ started \in [Hosts -> 0..MaxTickets]
          /\ pending \in [Hosts -> BOOLEAN] /\ published \in [Hosts -> 0..MaxTickets]
          /\ nextId \in 0..MaxRequests /\ slot \in 0..MaxRequests
          /\ requests \in [Ids -> [host : Hosts, ticket : 0..MaxTickets,
                status : {"unused", "running", "canceled", "success", "failure", "handled"},
                outcome : {"none", "canceled", "success", "failure"}]]
          /\ {badCleanup, crossCanceled, obsoleteHandled, lateIgnored, stoppedWaiting} \subseteq BOOLEAN
NoStalePublication == \A h \in Hosts : published[h] = 0 \/ published[h] = ticket[h]
NoObsoleteCleanup == ~badCleanup
CanceledOwnerSettled == \A i \in Ids :
  (requests[i].status = "handled" /\ requests[i].outcome = "canceled"
    /\ requests[i].ticket = ticket[requests[i].host]) => ~pending[requests[i].host]
RequestsHandled == \A i \in Ids : requests[i].status \in Awaiting ~> requests[i].status = "handled"
PendingSettles == \A h \in Hosts : pending[h] ~> ~pending[h]
NoCrossCancellation == ~crossCanceled
NoObsoleteHandler == ~obsoleteHandled
NoLateResponse == ~lateIgnored
NoStopWaiting == ~stoppedWaiting
=============================================================================
