import Lake
open Lake DSL

package "grt_collapse_freedom" where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.15.0"

@[default_target]
lean_lib «GRTCollapseFreedom» where
  roots := #[`GRTCollapseFreedom]
