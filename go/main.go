// Command gbm is the single entrypoint for the credit-risk Go services.
// It dispatches a subcommand to the inference or monitoring package:
//
//	gbm serve             scoring API (HTTP)
//	gbm drift             drift monitor (PSI/CSI)
//	gbm performance       performance monitor (AUC/KS/Gini)
//	gbm retrain [reason]  retrain orchestrator (default reason: manual)
//	gbm promote           atomically promote challenger to champion
//	gbm backfill          mature scoring_log outcomes from Gold labels
//	gbm sync              bulk feature-store upsert
//	gbm prune             delete scoring_log rows past retention
package main

import (
	"fmt"
	"os"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/inference"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/monitoring"
)

func usage() {
	fmt.Fprint(os.Stderr, `gbm - credit-risk Go services

usage: gbm <command> [args]

commands:
  serve              run the scoring API (HTTP)
  drift              run the drift monitor (PSI/CSI)
  performance        run the performance monitor (AUC/KS/Gini)
  retrain [reason]   run the retrain orchestrator (default reason: manual)
  promote            atomically promote the challenger to champion
  backfill           backfill scoring_log outcomes from Gold labels
  sync               sync Gold features to the feature store
  prune              delete scoring_log rows older than the retention window
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		inference.Serve()
	case "drift":
		monitoring.RunDrift()
	case "performance":
		monitoring.RunPerformance()
	case "retrain":
		reason := "manual"
		if len(os.Args) > 2 {
			reason = os.Args[2]
		}
		monitoring.RunRetrain(reason)
	case "promote":
		monitoring.RunPromote()
	case "backfill":
		monitoring.RunBackfill()
	case "sync":
		monitoring.RunSync()
	case "prune":
		monitoring.RunPrune()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}
