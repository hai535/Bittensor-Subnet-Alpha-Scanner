package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

//go:embed index.html
var indexHTML embed.FS

func main() {
	// Find a free port
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Println("Failed to find free port:", err)
		os.Exit(1)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	// Serve the viewer page
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, _ := indexHTML.ReadFile("index.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(data)
	})

	// API: list .md files in a directory
	http.HandleFunc("/api/list", func(w http.ResponseWriter, r *http.Request) {
		dir := r.URL.Query().Get("dir")
		if dir == "" {
			if len(os.Args) > 1 {
				dir = filepath.Dir(os.Args[1])
			} else {
				dir, _ = os.Getwd()
			}
		}

		var files []map[string]string
		entries, err := ioutil.ReadDir(dir)
		if err != nil {
			w.WriteHeader(500)
			fmt.Fprintf(w, "Error: %v", err)
			return
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".md") {
				files = append(files, map[string]string{
					"name": e.Name(),
					"path": filepath.Join(dir, e.Name()),
					"size": fmt.Sprintf("%.1f KB", float64(e.Size())/1024),
				})
			}
		}

		// Also list subdirectories
		var dirs []map[string]string
		for _, e := range entries {
			if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
				dirs = append(dirs, map[string]string{
					"name": e.Name(),
					"path": filepath.Join(dir, e.Name()),
				})
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"dir":   dir,
			"files": files,
			"dirs":  dirs,
		})
	})

	// API: read a .md file
	http.HandleFunc("/api/read", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Query().Get("path")
		if path == "" {
			w.WriteHeader(400)
			fmt.Fprint(w, "Missing path parameter")
			return
		}
		// Security: only allow .md files
		if !strings.HasSuffix(strings.ToLower(path), ".md") {
			w.WriteHeader(403)
			fmt.Fprint(w, "Only .md files allowed")
			return
		}
		data, err := ioutil.ReadFile(path)
		if err != nil {
			w.WriteHeader(404)
			fmt.Fprintf(w, "Error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write(data)
	})

	// API: get initial file (if passed as argument)
	http.HandleFunc("/api/init", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		result := map[string]string{}
		if len(os.Args) > 1 {
			absPath, _ := filepath.Abs(os.Args[1])
			result["file"] = absPath
		}
		json.NewEncoder(w).Encode(result)
	})

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	url := fmt.Sprintf("http://%s", addr)

	fmt.Printf("MD Viewer running at %s\n", url)
	fmt.Println("Press Ctrl+C to exit")

	// Open browser
	go openBrowser(url)

	if err := http.ListenAndServe(addr, nil); err != nil {
		fmt.Println("Server error:", err)
		os.Exit(1)
	}
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	cmd.Run()
}
