package monitoring

import (
	"strings"
	"testing"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

func TestSyncModelPath(t *testing.T) {
	if path, err := syncModelPath("champion"); err != nil || !strings.Contains(path, "champion") {
		t.Errorf("champion path = %q, err=%v", path, err)
	}
	if path, err := syncModelPath("challenger"); err != nil || !strings.Contains(path, "challenger") {
		t.Errorf("challenger path = %q, err=%v", path, err)
	}
	if _, err := syncModelPath("latest"); err == nil {
		t.Error("ambiguous model slot should fail")
	}
}

func TestValidateSyncContract(t *testing.T) {
	m := &model.Model{
		Version:        "v2.0",
		FeatureVersion: 2,
		Features:       []string{"a", "b"},
	}
	meta := &featureMetadata{
		FeatureVersion: 2,
		FeatureColumns: []string{"a", "b"},
	}
	if err := validateSyncContract(m, meta); err != nil {
		t.Fatalf("matching contract failed: %v", err)
	}

	badVersion := *meta
	badVersion.FeatureVersion = 3
	if err := validateSyncContract(m, &badVersion); err == nil {
		t.Error("feature-version mismatch should fail")
	}

	badColumns := *meta
	badColumns.FeatureColumns = []string{"b", "a"}
	if err := validateSyncContract(m, &badColumns); err == nil {
		t.Error("feature-column mismatch should fail")
	}
}

func TestValidateSyncContractSupportsLegacyVersion(t *testing.T) {
	m := &model.Model{Version: "legacy", Features: []string{"a"}}
	meta := &featureMetadata{FeatureVersion: 1, FeatureColumns: []string{"a"}}
	if err := validateSyncContract(m, meta); err != nil {
		t.Fatalf("legacy feature version should map to v1: %v", err)
	}
}
