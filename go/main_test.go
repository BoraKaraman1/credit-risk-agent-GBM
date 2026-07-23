package main

import "testing"

func TestSyncModelArg(t *testing.T) {
	for _, slot := range []string{"champion", "challenger"} {
		got, err := syncModelArg([]string{"--model", slot})
		if err != nil || got != slot {
			t.Errorf("syncModelArg(%q) = (%q, %v)", slot, got, err)
		}
	}
	for _, args := range [][]string{
		nil,
		{"challenger"},
		{"--model", "latest"},
		{"--model", "champion", "extra"},
	} {
		if _, err := syncModelArg(args); err == nil {
			t.Errorf("syncModelArg(%v) should fail", args)
		}
	}
}

func TestPromoteOfflineArg(t *testing.T) {
	if offline, err := promoteOfflineArg(nil); err != nil || offline {
		t.Errorf("promoteOfflineArg(nil) = (%v, %v)", offline, err)
	}
	if offline, err := promoteOfflineArg([]string{"--offline"}); err != nil || !offline {
		t.Errorf("promoteOfflineArg(--offline) = (%v, %v)", offline, err)
	}
	if _, err := promoteOfflineArg([]string{"--force"}); err == nil {
		t.Error("unknown promote flag should fail")
	}
}
