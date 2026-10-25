-------------------------- MODULE AutogradRouting --------------------------
EXTENDS Integers
CONSTANT Mutation
VARIABLES grad, data, hidden, tangent, wrapper, trajectory, surface, world, empty, pc, bounded, reject
vars == <<grad, data, hidden, tangent, wrapper, trajectory, surface, world, empty, pc, bounded, reject>>
Init == /\ grad \in BOOLEAN /\ data \in BOOLEAN /\ hidden \in BOOLEAN /\ tangent \in BOOLEAN
        /\ wrapper \in {"none", "vmap", "jvp"} /\ trajectory \in BOOLEAN /\ surface \in BOOLEAN
        /\ world \in {1, 2} /\ empty \in BOOLEAN
        /\ (hidden => wrapper # "none") /\ (tangent => wrapper = "jvp")
        /\ pc = 0 /\ bounded = FALSE /\ reject = FALSE
Records == data \/ hidden \/ tangent \/ trajectory \/ surface
Rejects == wrapper = "jvp" /\ trajectory /\ Mutation # "geometryJvp"
Route == /\ pc = 0 /\ pc' = 1
         /\ bounded' = (~Rejects /\ ~(Mutation = "empty" /\ empty) /\
                       ((grad /\ (world > 1 \/ data \/ trajectory \/ surface
                               \/ (Mutation # "shallow" /\ (hidden \/ tangent))))
                        \/ Mutation = "forceFunction"))
         /\ reject' = Rejects
         /\ UNCHANGED <<grad, data, hidden, tangent, wrapper, trajectory, surface, world, empty>>
Next == Route \/ (pc = 1 /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
BoundedRecording == pc = 1 /\ ~reject /\ grad /\ Records => bounded
PureForwardJvp == pc = 1 /\ world = 1 /\ wrapper = "jvp" /\ ~Records => ~bounded
GeometryForwardRejected == pc = 1 /\ wrapper = "jvp" /\ trajectory => reject
Termination == <>(pc = 1)
=============================================================================
