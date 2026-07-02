---
hide:
  - navigation
  - toc
---

<section class="heta-home" data-heta-home>
  <nav class="heta-home__nav" aria-label="Heta Framework home navigation">
    <a class="heta-home__brand" href="./">
      <img src="../images/heta-icon.png" alt="" />
      <span>Heta Framework</span>
    </a>
    <div class="heta-home__links">
      <a href="https://github.com/KnowledgeXLab/Heta_Framework">GitHub</a>
      <a href="https://knowledgexlab.github.io/">KnowledgeX Lab</a>
      <div class="heta-home__language-switch" aria-label="Language switcher">
        <a href="../" hreflang="zh">中文</a>
        <span aria-current="true">English</span>
      </div>
      <a class="heta-home__nav-cta" href="quick-start/">Quick Start</a>
    </div>
  </nav>

  <section class="heta-home__hero" aria-labelledby="heta-home-title">
    <div class="heta-home__hero-copy">
      <h1 id="heta-home-title">
        <span>Build the</span>
        <span>knowledge base</span>
        <span>you want with Heta.</span>
      </h1>
      <p class="heta-home__lead">
        Heta breaks knowledge-base construction into clear components: models, stores,
        parsers, steps, search modes, and benchmarks. Start with a simple vector KB,
        then add keyword retrieval, Heta-style graph knowledge, and evaluation when you need them.
      </p>
      <div class="heta-home__actions">
        <a class="heta-home__button heta-home__button--primary" href="quick-start/">
          Quick Start
        </a>
        <a class="heta-home__button" href="guides/what-is-recipe/">
          What is a Recipe?
        </a>
      </div>
    </div>

    <div class="heta-home__visual" aria-label="Recipe blocks assembled into a KnowledgeBase">
      <img
        class="heta-home__workflow-image"
        src="../images/home-recipe-to-kb.png"
        alt="Recipe uses models, stores, parsers, and steps to build a KnowledgeBase and unlock Search and Evaluate"
      />
      <svg class="heta-home__workflow-overlay" viewBox="0 0 1640 817" aria-hidden="true">
        <defs>
          <linearGradient id="heta-flow-gradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#7fc5c9" stop-opacity="0" />
            <stop offset="45%" stop-color="#2f9bac" stop-opacity="0.9" />
            <stop offset="100%" stop-color="#0b3e75" stop-opacity="0" />
          </linearGradient>
          <radialGradient id="heta-soft-aura" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#7fc5c9" stop-opacity="0.26" />
            <stop offset="58%" stop-color="#2f9bac" stop-opacity="0.08" />
            <stop offset="100%" stop-color="#2f9bac" stop-opacity="0" />
          </radialGradient>
        </defs>
        <ellipse class="heta-home__workflow-aura heta-home__workflow-aura--cluster" cx="1010" cy="390" rx="170" ry="125" />
        <ellipse class="heta-home__workflow-aura heta-home__workflow-aura--kb" cx="1388" cy="360" rx="190" ry="130" />
        <g class="heta-home__workflow-traces">
          <path d="M250 362 H372 V126 H790 C845 126 820 360 900 360" />
          <path d="M250 362 H372 V318 H790 C845 318 820 360 900 360" />
          <path d="M250 362 H372 V500 H790 C845 500 820 360 900 360" />
          <path d="M250 362 H372 V684 H790 C845 684 820 360 900 360" />
          <path d="M1128 362 H1235" />
        </g>
        <g class="heta-home__workflow-energy">
          <path style="--delay: 0s" d="M250 362 H372 V126 H790 C845 126 820 360 900 360" />
          <path style="--delay: 0.55s" d="M250 362 H372 V318 H790 C845 318 820 360 900 360" />
          <path style="--delay: 1.1s" d="M250 362 H372 V500 H790 C845 500 820 360 900 360" />
          <path style="--delay: 1.65s" d="M250 362 H372 V684 H790 C845 684 820 360 900 360" />
          <path style="--delay: 2.1s" d="M1128 362 H1235" />
        </g>
      </svg>
    </div>
  </section>

  <section class="heta-home__strip" aria-label="Heta core components">
    <span>Models</span>
    <span>Stores</span>
    <span>Parsers</span>
    <span>Steps</span>
    <span>Search</span>
    <span>Benchmarks</span>
  </section>

  <section class="heta-home__how" aria-labelledby="heta-how-title">
    <div class="heta-home__section-head heta-home__section-head--stacked">
      <p class="heta-home__eyebrow">How it works</p>
      <p id="heta-how-title" class="heta-home__section-intro">
        Heta does not ask you to write a complete RAG system all at once.
        You describe models, stores, parsers, and steps in a Recipe, and Heta builds a
        KnowledgeBase from that Recipe. After the build, the KB knows which search modes it
        supports and can be evaluated directly with benchmarks.
      </p>
    </div>

    <div class="heta-home__process" data-heta-process>
      <article class="heta-home__process-item">
        <figure class="heta-home__process-image">
          <img src="../images/home-recipe.png" alt="" loading="lazy" />
        </figure>
        <div class="heta-home__process-copy">
          <span>01</span>
          <h3>Recipe</h3>
          <p>
            A Recipe is the build plan for a knowledge base. It declares which model to use,
            where files are stored, which parsers are enabled, and which steps run in order.
            To reuse or change a KB, change the Recipe.
          </p>
        </div>
      </article>
      <article class="heta-home__process-item">
        <figure class="heta-home__process-image">
          <img src="../images/home-steps.png" alt="" loading="lazy" />
        </figure>
        <div class="heta-home__process-copy">
          <span>02</span>
          <h3>Steps</h3>
          <p>
            Steps perform the build: parse, split, embed, index, persist text, or build graph
            facts. Use only the steps needed for vector retrieval, or continue into the
            Heta graph procedure when relation-aware retrieval is needed.
          </p>
        </div>
      </article>
      <article class="heta-home__process-item">
        <figure class="heta-home__process-image">
          <img src="../images/home-search.png" alt="" loading="lazy" />
        </figure>
        <div class="heta-home__process-copy">
          <span>03</span>
          <h3>Search</h3>
          <p>
            Search works only with assets the KnowledgeBase has actually built. A vector index
            unlocks vector search, a text index unlocks full-text search, and graph assets unlock
            Heta graph search.
          </p>
        </div>
      </article>
      <article class="heta-home__process-item">
        <figure class="heta-home__process-image">
          <img src="../images/home-benchmark.png" alt="" loading="lazy" />
        </figure>
        <div class="heta-home__process-copy">
          <span>04</span>
          <h3>Benchmark</h3>
          <p>
            Benchmarks build KBs from the same Recipe, run query modes, and generate evaluation
            reports. This lets you compare Recipes with data instead of judging a single query by feel.
          </p>
        </div>
      </article>
    </div>
  </section>

  <section id="examples" class="heta-home__paths" aria-labelledby="heta-paths-title">
    <div class="heta-home__section-head heta-home__section-head--stacked">
      <p class="heta-home__eyebrow">Four cases</p>
      <p id="heta-paths-title" class="heta-home__section-intro">
        These cases use a local ObjectStore and in-memory stores. Models use common OpenAI
        LLM and embedding APIs by default. For Qwen, Milvus,
        PostgreSQL, or Elasticsearch, replace only the corresponding component.
      </p>
    </div>

    <div class="heta-home__playground" data-heta-code-tabs>
      <aside class="heta-home__playground-nav" aria-label="Choose a case">
        <button class="heta-home__case-tab is-active" type="button" role="tab" aria-selected="true"
          data-heta-code-tab="vector" data-title="examples/home_vector_case.en.py">
          <span>01</span>
          <strong>Vector KB</strong>
        </button>
        <button class="heta-home__case-tab" type="button" role="tab" aria-selected="false"
          data-heta-code-tab="full-text" data-title="examples/home_full_text_case.en.py">
          <span>02</span>
          <strong>Keyword retrieval KB</strong>
        </button>
        <button class="heta-home__case-tab" type="button" role="tab" aria-selected="false"
          data-heta-code-tab="graph" data-title="examples/home_graph_case.en.py">
          <span>03</span>
          <strong>Heta graph KB</strong>
        </button>
        <button class="heta-home__case-tab" type="button" role="tab" aria-selected="false"
          data-heta-code-tab="benchmark" data-title="examples/home_benchmark_case.en.py">
          <span>04</span>
          <strong>Benchmark evaluation</strong>
        </button>
      </aside>

      <div class="heta-home__terminal">
        <div class="heta-home__terminal-bar">
          <div class="heta-home__terminal-dots" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="heta-home__terminal-title" data-heta-terminal-title>
            examples/home_vector_case.en.py
          </div>
        </div>

        <div class="heta-home__terminal-body">
          <article class="heta-home__terminal-panel is-active" data-heta-code-panel="vector" markdown="1">
          <div class="heta-home__run-command">
            <span>$</span>
            <code>OPENAI_API_KEY=... PYTHONPATH=src python docs/examples/home_vector_case.en.py</code>
          </div>

```python
--8<-- "docs/examples/home_vector_case.en.py"
```
          <div class="heta-home__terminal-output">
            <span>Example output</span>
            <pre><code>Heta builds a knowledge base by creating KnowledgeBase objects from Recipe definitions [1].
Heta builds KnowledgeBase objects from Recipe definitions. Vector search retrieves chunks by semantic similarity.</code></pre>
          </div>
        </article>

        <article class="heta-home__terminal-panel" data-heta-code-panel="full-text" markdown="1">
          <div class="heta-home__run-command">
            <span>$</span>
            <code>OPENAI_API_KEY=... PYTHONPATH=src python docs/examples/home_full_text_case.en.py</code>
          </div>

```python
--8<-- "docs/examples/home_full_text_case.en.py"
```
          <div class="heta-home__terminal-output">
            <span>Example output</span>
            <pre><code>BM25-style retrieval is useful for exact terms and identifiers [1].
Heta can add full-text search with IndexFullText. BM25-style retrieval is useful for exact terms and identifiers.</code></pre>
          </div>
        </article>

        <article class="heta-home__terminal-panel" data-heta-code-panel="graph" markdown="1">
          <div class="heta-home__run-command">
            <span>$</span>
            <code>OPENAI_API_KEY=... PYTHONPATH=src python docs/examples/home_graph_case.en.py</code>
          </div>

```python
--8<-- "docs/examples/home_graph_case.en.py"
```
          <div class="heta-home__terminal-output">
            <span>Example output</span>
            <pre><code>Heta creates a KnowledgeBase by building it from recipes [1][2][3].
relation Relation: Heta -> KnowledgeBase
Name: builds
Type: creates
Description: Heta builds knowledge bases from recipes.</code></pre>
          </div>
        </article>

        <article class="heta-home__terminal-panel" data-heta-code-panel="benchmark" markdown="1">
          <div class="heta-home__run-command">
            <span>$</span>
            <code>OPENAI_API_KEY=... PYTHONPATH=src python docs/examples/home_benchmark_case.en.py</code>
          </div>

```python
--8<-- "docs/examples/home_benchmark_case.en.py"
```
          <div class="heta-home__terminal-output">
            <span>Example output</span>
            <pre><code>{'vector_search.evidence_recall@1': 1.0}
_heta/knowledge_bases/home-benchmark/evaluations/home_demo/report.json</code></pre>
          </div>
        </article>
        </div>
      </div>
    </div>
  </section>

  <section class="heta-home__case" aria-labelledby="heta-case-title">
    <div>
      <p class="heta-home__eyebrow">Benchmark support</p>
      <p id="heta-case-title" class="heta-home__section-intro heta-home__section-intro--narrow">
        A benchmark adapter prepares data, builds KnowledgeBases, runs query modes, and generates
        evaluation reports. Use built-in benchmarks or implement the protocol for your own business eval set.
      </p>
    </div>
    <div class="heta-home__case-grid">
      <div>
        <strong>
          <a href="https://github.com/yixuantt/MultiHop-RAG" target="_blank" rel="noopener">
            MultiHop-RAG
          </a>
        </strong>
        <span>
          A multi-hop QA benchmark for complex queries, evidence recall, and multi-hop search.
          <a href="https://huggingface.co/datasets/yixuantt/MultiHopRAG" target="_blank" rel="noopener">Dataset</a>
        </span>
      </div>
      <div>
        <strong>
          <a href="https://github.com/beir-cellar/beir" target="_blank" rel="noopener">
            BEIR
          </a>
        </strong>
        <span>
          A standard information retrieval benchmark. Heta currently supports SciFact, NFCorpus, FiQA, and HotpotQA.
          <a href="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/" target="_blank" rel="noopener">Datasets</a>
        </span>
      </div>
      <div>
        <strong>
          <a href="https://github.com/qinchuanhui/UDA-Benchmark" target="_blank" rel="noopener">
            UDA-Benchmark
          </a>
        </strong>
        <span>
          A real document-analysis benchmark that can build multiple KBs per case to evaluate different Recipes.
          <a href="https://huggingface.co/datasets/qinchuanhui/UDA-QA" target="_blank" rel="noopener">Source documents</a>
        </span>
      </div>
    </div>
  </section>
</section>
