#!/bin/bash -l
#SBATCH --job-name=bench3d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --output=bench_%x_%j.out
#SBATCH --error=bench_%x_%j.err

# Usage:
#   sbatch --export=ALL,CASE=cbox,SWEEP=h       job_bench3d.sh
#   sbatch --export=ALL,CASE=monolith,SWEEP=h   job_bench3d.sh
#   sbatch --export=ALL,CASE=cbox,SWEEP=check   job_bench3d.sh    # 2D-equivalence test (cheap)
#   sbatch --export=ALL,CASE=rotbox,SWEEP=h     job_bench3d.sh    # clamped box, left end twisted by 180 deg
#   sbatch --export=ALL,CASE=rotbox,SWEEP=h,ROT_DEG=90 job_bench3d.sh
# CASE  = cbox | monolith | rotbox
# SWEEP = check | h | p | hp      (check is NOT available for rotbox: a twist has no 2D equivalent)
# Optional overrides (comma-separated after ALL,):
#   MATERIAL_MODEL (cbox default neo_hooke, monolith/rotbox default hartmann_neff), REG_VALUE (1e-5),
#   REG_METHODS (huhu_lulu,huhu), DEPTH, NZ, ZBC (free|plane_strain), CELL_TYPE, MESHES, P_VALUES
# rotbox only:
#   ROT_DEG (twist angle of the left end, default 180), LAYER (1/0, enclosing third-medium layer, default 1),
#   LAYER_Z (1/0, layer also in z, default 1), OUTER_BC (free|fixed, default free)

. /etc/profile.d/module.sh
module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

case "$CASE" in
  cbox|monolith|rotbox) ;;
  *) echo "CASE must be cbox, monolith or rotbox (got '$CASE')"; exit 1 ;;
esac

# ---- sweep definitions: edit the values here ----
# MESHES uses the 2D definition NxxNy (cbox: Nx%10==0, Ny%5==0; monolith/rotbox: Nx%20==0, Ny%5==0).
# Elements over the depth: NZ (default 8). p must be >= 2.
# dofs at p=2 (NZ=8):  cbox 20x10 48k | 40x20 186k | 60x30 414k ;  monolith 20x10 44k | 40x20 169k | 60x30 376k
# rotbox has one extra element layer on every side (Nx+2, Ny+2, NZ+2 elements), so it is somewhat larger than monolith.
# p=3 is ~3.2x larger.
case "$SWEEP" in
  check) export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2}";   export ZBC=plane_strain; export NZ=1 ;;
  h)     export MESHES="${MESHES:-20x10,40x20,60x30}"; export P_VALUES="${P_VALUES:-2}" ;;
  p)     export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2,3}" ;;
  hp)    export MESHES="${MESHES:-20x10,40x20}";       export P_VALUES="${P_VALUES:-2,3}" ;;
  *) echo "SWEEP must be check, h, p or hp (got '$SWEEP')"; exit 1 ;;
esac

if [ "$CASE" = "rotbox" ]; then
  if [ "$SWEEP" = "check" ] || [ "${ZBC:-free}" = "plane_strain" ]; then
    echo "CASE=rotbox needs ZBC=free and has no 2D-equivalence check (use SWEEP=h, p or hp)"; exit 1
  fi
  export ROT_DEG="${ROT_DEG:-180}"
  export LAYER="${LAYER:-1}"
  export LAYER_Z="${LAYER_Z:-1}"
  export OUTER_BC="${OUTER_BC:-free}"
fi

echo "node: $(hostname)  kernel: $(uname -r)  glibc: $(ldd --version | head -1)"
echo "CASE=$CASE SWEEP=$SWEEP MESHES=$MESHES P_VALUES=$P_VALUES ZBC=${ZBC:-free} NZ=${NZ:-8} DEPTH=${DEPTH:-<2D width>}"
echo "MATERIAL_MODEL=${MATERIAL_MODEL:-<case default>} REG_VALUE=${REG_VALUE:-1e-5} REG_METHODS=${REG_METHODS:-huhu_lulu,huhu}"
if [ "$CASE" = "rotbox" ]; then
  echo "ROT_DEG=$ROT_DEG LAYER=$LAYER LAYER_Z=$LAYER_Z OUTER_BC=$OUTER_BC"
fi

MYWORKDIR=$SCRATCH/$SLURM_JOBID
mkdir -p "$MYWORKDIR"
cp "$SLURM_SUBMIT_DIR"/*.py "$MYWORKDIR"/
cd "$MYWORKDIR"
export XDG_CACHE_HOME="$MYWORKDIR/cache"; export HOME="$MYWORKDIR"; mkdir -p "$XDG_CACHE_HOME"

RESULTDIR=$SLURM_SUBMIT_DIR/results3d_${CASE}_${SWEEP}_$SLURM_JOBID
save_results() {
    mkdir -p "$RESULTDIR"
    cp -r ${CASE}3d_* "$RESULTDIR"/ 2>/dev/null
    cd "$SLURM_SUBMIT_DIR" && rm -rf "$MYWORKDIR"
}
trap save_results EXIT

mpirun -n "$SLURM_NTASKS" python ${CASE}_3d.py