-------------------------- MODULE BudgetAgreement --------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Ranks, Mutation
VARIABLES free, dataFree, cuda, cudaData, hostData, hostFree, explicit, localLeading,
          devicePlan, hostPlan, leadingPlan, pc
vars == <<free, dataFree, cuda, cudaData, hostData, hostFree, explicit, localLeading,
          devicePlan, hostPlan, leadingPlan, pc>>
Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Max(a, b) == IF a >= b THEN a ELSE b
\* free/dataFree are _memory_lower_bounds(...)[0], already retaining 3/4.
\* Zero denotes the fixed CPU plan or an absent host limit, not zero bytes.
LocalDevice(r) == IF ~cuda[r] THEN 0 ELSE
                  Min({free[r]} \cup (IF cudaData[r] THEN {dataFree[r]} ELSE {})) \div 3
LocalHost(r) == IF hostData[r] THEN Max(8, Min({hostFree[r] \div 4, LocalDevice(r)})) ELSE 0
SharedDevice == IF explicit \/ (\E r \in Ranks : ~cuda[r]) THEN 0
                ELSE Min({LocalDevice(r) : r \in Ranks})
WithHost == {r \in Ranks : hostData[r]}
SharedHost == IF SharedDevice = 0 \/ WithHost = {} THEN 0
              ELSE Min({LocalHost(r) : r \in WithHost})
Init == /\ free \in [Ranks -> {12, 96}] /\ dataFree \in [Ranks -> {6, 48}]
        /\ cuda \in [Ranks -> BOOLEAN] /\ cudaData \in [Ranks -> BOOLEAN]
        /\ hostData \in [Ranks -> BOOLEAN] /\ hostFree \in [Ranks -> {32, 64}]
        /\ explicit \in BOOLEAN /\ localLeading \in [Ranks -> {1, 2}]
        /\ devicePlan = [r \in Ranks |-> 0] /\ hostPlan = [r \in Ranks |-> 0]
        /\ leadingPlan = [r \in Ranks |-> 0] /\ pc = 0
Agree == /\ pc = 0 /\ pc' = 1
         /\ devicePlan' = [r \in Ranks |->
               CASE Mutation = "device" -> LocalDevice(r)
                 [] Mutation = "deviceMax" -> Max(8, SharedDevice)
                 [] OTHER -> SharedDevice]
         /\ hostPlan' = [r \in Ranks |->
               IF Mutation = "host" THEN LocalHost(r)
               ELSE IF Mutation = "hostMax" THEN Max(16, SharedHost) ELSE SharedHost]
         /\ UNCHANGED <<free, dataFree, cuda, cudaData, hostData, hostFree, explicit,
                        localLeading, leadingPlan>>
AgreeLeading == /\ pc = 1 /\ pc' = 2
                /\ leadingPlan' = [r \in Ranks |-> IF Mutation = "localLeading"
                       THEN localLeading[r] ELSE Min({localLeading[s] : s \in Ranks})]
                /\ UNCHANGED <<free, dataFree, cuda, cudaData, hostData, hostFree, explicit,
                               localLeading, devicePlan, hostPlan>>
Next == Agree \/ AgreeLeading \/ (pc = 2 /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
DeviceAgreement == Cardinality({devicePlan[r] : r \in Ranks}) = 1
HostAgreement == Cardinality({hostPlan[r] : r \in Ranks}) = 1
DeviceMinimum == pc > 0 => \A r \in Ranks : devicePlan[r] = SharedDevice
HostMinimum == pc > 0 => \A r \in Ranks : hostPlan[r] = SharedHost
LeadingAgreement == pc = 2 => \A r \in Ranks : leadingPlan[r] = Min({localLeading[s] : s \in Ranks})
Termination == <>(pc = 2)
=============================================================================
