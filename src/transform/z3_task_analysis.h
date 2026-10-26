/*!
 * \file z3_task_analysis.h
 * \brief Opt-in task and cost inference for CUDA Z3 pipeline loops.
 */
#ifndef TVM_TL_TRANSFORM_Z3_TASK_ANALYSIS_H_
#define TVM_TL_TRANSFORM_Z3_TASK_ANALYSIS_H_

#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief Complete missing CUDA Z3 solver inputs from scheduled loop statements.
 *
 * The pass is disabled unless the PassContext config
 * ``tl.enable_z3_task_analysis`` is true. It only visits loops carrying
 * ``tl_z3_pipeline`` and never replaces an explicitly present solver field.
 * Statement extraction deliberately mirrors Z3PipelinePlanning: normalize the
 * top-level body, recursively flatten SeqStmt, omit buffer declarations and
 * replayable scalar Bind statements, and retain source order.
 *
 * Cost inference requires these loop annotations:
 *  - ``tl_z3_pipeline_hardware``: hardware-spec JSON path;
 *  - ``tl_z3_pipeline_calibration``: latency-calibration JSON path;
 *  - optional ``tl_z3_pipeline_require_measured`` (default false).
 *
 * The Python module ``tilelang.cuda.task_cost`` must be imported so its
 * ``tl.cuda.estimate_task`` and ``tl.cuda.z3_memory_group_capacities`` FFI
 * callbacks are registered. Unsupported or opaque analyses fail explicitly;
 * no default hardware capacity, zero byte count, or dropped dependency is
 * substituted.
 */
TVM_DLL tvm::transform::Pass InferZ3PipelineProblem();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_Z3_TASK_ANALYSIS_H_
