package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type testSemanticEmbedder struct{}

func (testSemanticEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		text = strings.ToLower(text)
		if strings.Contains(text, "write") || strings.Contains(text, "save") {
			out[i] = []float32{1, 0}
		} else {
			out[i] = []float32{0, 1}
		}
	}
	return out, nil
}

func TestParseSemanticDescriptions(t *testing.T) {
	got, err := parseSemanticDescriptions([]byte("Here is the result:\n[{\"id\":1,\"description\":\" reads data \"},{\"id\":0,\"description\":\"writes data\"}]\n"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != "writes data" || got[1] != "reads data" {
		t.Fatalf("descriptions = %#v", got)
	}
	if _, err := parseSemanticDescriptions([]byte("[]"), 1); err == nil {
		t.Fatal("expected missing description to fail")
	}
}

func TestReadOnlyAnalysisArgs(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want []string
		omit []string
	}{
		{
			name: "claude",
			args: []string{"claude", "--permission-mode", "acceptEdits", "-p", "summarize"},
			want: []string{"--permission-mode", "plan"},
		},
		{
			name: "gemini",
			args: []string{"gemini", "--approval-mode", "auto_edit", "-p", "summarize"},
			want: []string{"--approval-mode", "plan"},
		},
		{
			name: "codex",
			args: []string{"codex", "exec", "--approve-for-me", "--sandbox", "workspace-write", "--skip-git-repo-check", "summarize"},
			want: []string{"--sandbox", "read-only", "--skip-git-repo-check"},
			omit: []string{"--approve-for-me", "workspace-write"},
		},
		{
			name: "codex legacy approval flag",
			args: []string{"codex", "exec", "--ask-for-approval", "never", "summarize"},
			want: []string{"--sandbox", "read-only"},
			omit: []string{"--ask-for-approval", "never"},
		},
		{
			name: "aider",
			args: []string{"aider", "--yes-always", "--message", "summarize"},
			want: []string{"--dry-run"},
			omit: []string{"--yes-always"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := readOnlyAnalysisArgs(tt.args)
			joined := strings.Join(got, " ")
			for _, flag := range tt.want {
				if !strings.Contains(joined, flag) {
					t.Errorf("args %v missing %q", got, flag)
				}
			}
			for _, flag := range tt.omit {
				if strings.Contains(joined, flag) {
					t.Errorf("args %v unexpectedly contain %q", got, flag)
				}
			}
		})
	}
}

func TestCollectSemanticMethods(t *testing.T) {
	root := t.TempDir()
	source := "package demo\n\nfunc SaveFile(path string) error {\n\treturn os.WriteFile(path, nil, 0600)\n}\n\nfunc LoadFile(path string) error {\n\treturn nil\n}\n"
	if err := os.WriteFile(filepath.Join(root, "files.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	methods, err := collectSemanticMethods(root, []FileEntry{{Path: "files.go", Size: int64(len(source))}})
	if err != nil {
		t.Fatal(err)
	}
	if len(methods) != 2 {
		t.Fatalf("found %d methods, want 2: %#v", len(methods), methods)
	}
	if methods[0].Name != "SaveFile" || methods[0].Line != 3 || !strings.Contains(methods[0].Source, "WriteFile") {
		t.Fatalf("first method = %#v", methods[0])
	}
	if methods[1].Name != "LoadFile" || !strings.Contains(methods[1].Source, "func LoadFile") {
		t.Fatalf("second method = %#v", methods[1])
	}
}

func TestSemanticManagerIndexesAndSearches(t *testing.T) {
	isolateSettings(t)
	root := t.TempDir()
	source := "package demo\n\nfunc SaveFile(path string) error {\n\treturn nil\n}\n\nfunc ReadConfig(path string) error {\n\treturn nil\n}\n"
	if err := os.WriteFile(filepath.Join(root, "methods.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	harness := writeHarness(t, `printf '%s\n' '[{"id":0,"description":"writes and saves a file to disk"},{"id":1,"description":"reads and parses configuration data"}]'`+"\n")
	agent, err := newAgentManager(root, harness+" {prompt}", nil)
	if err != nil {
		t.Fatal(err)
	}
	ix := NewIndex(root)
	ix.Build()
	manager := newSemanticManagerWithEmbedder(ix, agent, testSemanticEmbedder{})
	if err := manager.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		status := manager.Status(true)
		if status.State == "error" {
			t.Fatalf("indexing failed: %s", status.Error)
		}
		if status.State == "ready" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if status := manager.Status(true); status.State != "ready" || status.Indexed != 2 {
		t.Fatalf("status = %+v, want ready with two methods", status)
	}
	hits, err := manager.Search(context.Background(), "how do I save a file", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Name != "SaveFile" {
		t.Fatalf("semantic hits = %#v, want SaveFile first", hits)
	}
}

func TestOpenAICompatibleEmbedder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
		}
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}`))
	}))
	defer server.Close()
	embedder := &openAICompatibleEmbedder{endpoint: server.URL + "/v1/embeddings", model: "test-model", apiKey: "test-key", client: server.Client()}
	vectors, err := embedder.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 2 || vectors[0][0] != 1 || vectors[1][1] != 1 {
		t.Fatalf("vectors = %#v", vectors)
	}
}

func TestSemanticSearchSettingDefaultsOff(t *testing.T) {
	for _, item := range settingsSchema {
		if item.Key == "semantic.enabled" {
			if item.Default != false {
				t.Fatalf("semantic.enabled default = %v, want false", item.Default)
			}
			return
		}
	}
	t.Fatal("semantic.enabled setting is missing from the schema")
}