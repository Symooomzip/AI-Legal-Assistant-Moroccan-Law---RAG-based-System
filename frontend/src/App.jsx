import { useMemo, useState } from "react";
import { analyzeDocument, healthCheck, sendChatMessage } from "./api";

const SECTIONS = {
  CHAT: "consultation",
  DOCUMENT: "document",
  ABOUT: "about"
};

const SUGGESTED_PROMPTS = [
  "Quelle est la procédure de divorce pour discorde selon la Moudawana ?",
  "ما هي عقوبة السرقة الموصوفة في القانون الجنائي المغربي؟",
  "Expliquez l'article 16 de la Constitution marocaine.",
  "Quels sont les droits de garde après divorce ?",
  "ما هي شروط صحة العقد في القانون المدني المغربي؟"
];

function App() {
  const [section, setSection] = useState(SECTIONS.CHAT);
  const [status, setStatus] = useState("Unknown");
  const [sessionId] = useState(() => `session_${Date.now()}`);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const [documentFile, setDocumentFile] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);

  const navItems = useMemo(
    () => [
      { id: SECTIONS.CHAT, label: "Consultation" },
      { id: SECTIONS.DOCUMENT, label: "Document Review" },
      { id: SECTIONS.ABOUT, label: "About" }
    ],
    []
  );

  async function onHealthCheck() {
    try {
      setStatus("Checking...");
      const payload = await healthCheck();
      setStatus(payload.status || "ok");
    } catch (error) {
      setStatus(`Unavailable (${error.message})`);
    }
  }

  async function onSubmitChat(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || chatLoading) {
      return;
    }

    setChatError("");
    setChatLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuestion("");

    try {
      const payload = await sendChatMessage(trimmed, sessionId);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: payload.answer || "No answer returned." }
      ]);
    } catch (error) {
      setChatError(error.message);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "The request failed. Please try again." }
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  function onClearConversation() {
    setMessages([]);
    setChatError("");
    setQuestion("");
  }

  function onSelectPrompt(prompt) {
    setQuestion(prompt);
    setChatError("");
  }

  async function onAnalyzeDocument(event) {
    event.preventDefault();
    if (!documentFile || analysisLoading) {
      return;
    }

    setAnalysisLoading(true);
    setAnalysisError("");
    setAnalysisResult(null);

    try {
      const payload = await analyzeDocument(documentFile, sessionId);
      setAnalysisResult(payload);
    } catch (error) {
      setAnalysisError(error.message);
    } finally {
      setAnalysisLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Moroccan Legal Assistant</h1>
          <p>Professional legal consultation and document intelligence.</p>
        </div>
        <div className="status-panel">
          <span>API Status: {status}</span>
          <button className="btn btn-secondary" type="button" onClick={onHealthCheck}>
            Check API
          </button>
        </div>
      </header>

      <nav className="nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item ${section === item.id ? "active" : ""}`}
            onClick={() => setSection(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main className="main">
        {section === SECTIONS.CHAT && (
          <section className="panel">
            <h2>Consultation</h2>
            <div className="chat-window">
              {messages.length === 0 && <p className="muted">Start by asking a legal question in French or Arabic.</p>}
              {messages.map((message, index) => (
                <article key={index} className={`bubble ${message.role}`}>
                  <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>

            <form className="chat-form" onSubmit={onSubmitChat}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Type your legal question here..."
                rows={4}
              />
              <div className="suggested-prompts">
                <p className="suggested-prompts-title">Suggested Prompts</p>
                <div className="suggested-prompts-grid">
                  {SUGGESTED_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="prompt-chip"
                      onClick={() => onSelectPrompt(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
              <div className="actions">
                <button className="btn btn-primary" type="submit" disabled={chatLoading}>
                  {chatLoading ? "Sending..." : "Submit Question"}
                </button>
                <button className="btn btn-secondary" type="button" onClick={onClearConversation}>
                  Clear
                </button>
              </div>
            </form>

            {chatError && <p className="error">{chatError}</p>}
          </section>
        )}

        {section === SECTIONS.DOCUMENT && (
          <section className="panel">
            <h2>Document Review</h2>
            <form className="doc-form" onSubmit={onAnalyzeDocument}>
              <input
                type="file"
                accept=".pdf"
                onChange={(event) => setDocumentFile(event.target.files?.[0] || null)}
              />
              <button className="btn btn-primary" type="submit" disabled={!documentFile || analysisLoading}>
                {analysisLoading ? "Analyzing..." : "Analyze Document"}
              </button>
            </form>

            {analysisError && <p className="error">{analysisError}</p>}

            {analysisResult && (
              <div className="analysis-box">
                <h3>{analysisResult.filename}</h3>
                <p>
                  Pages: {analysisResult.metadata?.total_pages || "N/A"} | Language:{" "}
                  {analysisResult.metadata?.language || "N/A"}
                </p>
                <h4>Summary</h4>
                <pre>{analysisResult.summary || "No summary returned."}</pre>
              </div>
            )}
          </section>
        )}

        {section === SECTIONS.ABOUT && (
          <section className="panel">
            <h2>About</h2>
            <p>
              This platform provides structured legal information retrieval for Moroccan law.
              Responses should always be validated by a licensed legal professional before any legal action.
            </p>
            <ul>
              <li>Hybrid retrieval and article-aware legal referencing</li>
              <li>Bilingual support in French and Arabic</li>
              <li>Legal document analysis workflow</li>
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
