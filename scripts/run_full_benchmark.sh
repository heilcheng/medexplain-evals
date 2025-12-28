#!/bin/bash
#
# MEQ-Bench 2.0 - Full Benchmark Run Script
#
# This script orchestrates a complete benchmark evaluation:
# 1. Validate environment
# 2. Curate dataset (if needed)
# 3. Generate explanations
# 4. Compute scores
# 5. Generate analysis reports
#
# Usage:
#   ./scripts/run_full_benchmark.sh [options]
#
# Options:
#   --models MODEL1,MODEL2    Models to evaluate (comma-separated)
#   --items N                 Number of items to evaluate (default: all)
#   --output DIR              Output directory (default: results/)
#   --smoke-test              Run quick smoke test (10 items, 1 model)
#   --local-only              Only use locally-hosted models (Llama)
#   --skip-curation           Skip dataset curation step
#   --skip-generation         Skip explanation generation
#   --skip-scoring            Skip score computation
#   --dry-run                 Show what would be done without executing
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODELS="gpt-4o"
ITEMS=""
OUTPUT_DIR="results"
SMOKE_TEST=false
LOCAL_ONLY=false
SKIP_CURATION=false
SKIP_GENERATION=false
SKIP_SCORING=false
DRY_RUN=false
BENCHMARK_PATH="data/benchmark_v2/test.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --items)
            ITEMS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --skip-curation)
            SKIP_CURATION=true
            shift
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip-scoring)
            SKIP_SCORING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Configure for smoke test
if [ "$SMOKE_TEST" = true ]; then
    echo -e "${YELLOW}Running in SMOKE TEST mode${NC}"
    ITEMS=10
    MODELS="gpt-4o"
fi

# Configure for local-only
if [ "$LOCAL_ONLY" = true ]; then
    echo -e "${YELLOW}Running in LOCAL-ONLY mode (Llama models)${NC}"
    MODELS="llama-4-scout"
fi

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Utility functions
log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute: $*${NC}"
    else
        "$@"
    fi
}

# Header
echo -e "\n${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      MEQ-BENCH 2.0 - FULL BENCHMARK RUNNER         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}\n"

echo "Configuration:"
echo "  Models:      $MODELS"
echo "  Items:       ${ITEMS:-all}"
echo "  Output:      $OUTPUT_DIR"
echo "  Benchmark:   $BENCHMARK_PATH"

# Step 1: Validate Environment
log_step "Step 1: Validating Environment"

if [ "$DRY_RUN" = false ]; then
    python scripts/validate_environment.py --quick
    if [ $? -ne 0 ]; then
        log_error "Environment validation failed"
        exit 1
    fi
    log_success "Environment validated"
else
    echo "[DRY RUN] Would run: python scripts/validate_environment.py --quick"
fi

# Step 2: Dataset Curation
if [ "$SKIP_CURATION" = false ]; then
    log_step "Step 2: Dataset Curation"
    
    if [ -f "$BENCHMARK_PATH" ]; then
        log_success "Benchmark dataset exists: $BENCHMARK_PATH"
    else
        log_warning "Benchmark dataset not found, running curation..."
        run_cmd python scripts/curate_dataset.py \
            --output "data/benchmark_v2/full_dataset.json" \
            --sample-only
        log_success "Dataset curation complete"
    fi
else
    log_step "Step 2: Dataset Curation (SKIPPED)"
fi

# Step 3: Cost Estimation
log_step "Step 3: Cost Estimation"

ITEM_COUNT=${ITEMS:-100}  # Default to 100 for estimation
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
MODEL_ARGS=""
for model in "${MODEL_ARRAY[@]}"; do
    MODEL_ARGS="$MODEL_ARGS $model"
done

run_cmd python scripts/estimate_cost.py --models $MODEL_ARGS --items "$ITEM_COUNT"

if [ "$DRY_RUN" = false ]; then
    read -p "Continue with evaluation? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Step 4: Run Evaluation
log_step "Step 4: Running Evaluation"

EVAL_ARGS="--benchmark $BENCHMARK_PATH --output $OUTPUT_DIR"

# Add models
EVAL_ARGS="$EVAL_ARGS --models"
for model in "${MODEL_ARRAY[@]}"; do
    EVAL_ARGS="$EVAL_ARGS $model"
done

# Add item limit if specified
if [ -n "$ITEMS" ]; then
    EVAL_ARGS="$EVAL_ARGS --max-items $ITEMS"
fi

# Add skip flags
if [ "$SKIP_GENERATION" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --skip-generation"
fi

if [ "$SKIP_SCORING" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --skip-scoring"
fi

run_cmd python scripts/run_evaluation.py $EVAL_ARGS

log_success "Evaluation complete"

# Step 5: Generate Reports
log_step "Step 5: Generating Analysis Reports"

# Create analysis runner
if [ "$DRY_RUN" = false ]; then
    python -c "
import sys
sys.path.insert(0, '.')
from analysis import ScoreAnalyzer, MEQBenchVisualizer, ReportGenerator, ErrorAnalyzer, StatisticalTests

# Load and analyze
analyzer = ScoreAnalyzer('$OUTPUT_DIR')
analyzer.load_scores()
results = analyzer.analyze()

# Visualizations
viz = MEQBenchVisualizer('reports/figures')
viz.generate_all_figures(results.to_dict())

# Error analysis
error_analyzer = ErrorAnalyzer()
if analyzer.scores_df is not None:
    error_analyzer.load_data(analyzer.scores_df)
    error_report = error_analyzer.run_full_analysis()
else:
    error_report = None

# Statistical tests
stats_tester = StatisticalTests()
if analyzer.scores_df is not None and len(analyzer.scores_df['model'].unique()) > 1:
    stats_results = stats_tester.run_comprehensive_analysis(analyzer.scores_df)
else:
    stats_results = None

# Generate reports
reporter = ReportGenerator('reports')
reporter.generate_all_reports(
    analysis_results=results.to_dict(),
    error_report=error_report.to_dict() if error_report else None,
    statistical_results=stats_results,
    figures_dir='reports/figures'
)

print('Reports generated successfully!')
"
    log_success "Reports generated"
else
    echo "[DRY RUN] Would generate analysis reports"
fi

# Summary
log_step "Summary"

echo -e "${GREEN}Benchmark run completed successfully!${NC}\n"
echo "Results saved to:"
echo "  - Scores:       $OUTPUT_DIR/"
echo "  - Figures:      reports/figures/"
echo "  - HTML Report:  reports/summary_report.html"
echo "  - Markdown:     reports/summary_report.md"
echo ""

if [ "$SMOKE_TEST" = true ]; then
    echo -e "${YELLOW}Note: This was a SMOKE TEST with limited items.${NC}"
    echo "For full evaluation, run without --smoke-test"
fi

echo -e "\n${GREEN}Done!${NC}\n"

