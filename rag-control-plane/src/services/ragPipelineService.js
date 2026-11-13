/**
 * NeuroMed RAG Pipeline Engine & Service Layer
 * Enterprise Simulation of Document Ingestion, Dynamic Chunking, Vector Storage, and Semantic Retrieval
 */

export const INITIAL_DOCUMENTS = [
  {
    id: "doc-easl-2024",
    title: "EASL Clinical Practice Guidelines: Wilson's Disease (2024 Update)",
    filename: "EASL_Wilson_Disease_2024_Guidelines.pdf",
    category: "Hepatology",
    format: "PDF",
    fileSize: "4.8 MB",
    uploadDate: "2026-10-14",
    status: "Active",
    chunkCount: 142,
    tokenCount: 72400,
    usageCount: 1842,
    avgLatencyMs: 38,
    similarityThreshold: 0.82,
    accessPermission: "Public CDSS",
    description: "Official European Association for the Study of the Liver consensus on early diagnosis, ceruloplasmin cutoffs, and 24h urinary copper excretion standards.",
    content: `Wilson's disease (WD) is an autosomal recessive disorder of hepatic copper disposition caused by mutations in the ATP7B gene on chromosome 13q14.3. Impaired biliary excretion leads to systemic copper accumulation primarily affecting the liver and central nervous system. 

Diagnostic confirmation requires integrating clinical phenotypes, biochemical copper indices, slit-lamp examination for Kayser-Fleischer (KF) rings, and direct ATP7B sequencing. Serum ceruloplasmin below 20 mg/dL (0.2 g/L) demonstrates 90% diagnostic sensitivity, though normal levels do not exclude disease due to acute-phase reactant elevation.

Basal 24-hour urinary copper excretion exceeding 100 µg/day (1.6 µmol/day) in symptomatic patients or >40 µg/day in asymptomatic siblings is diagnostic. First-line pharmacotherapy for symptomatic hepatic disease consists of copper chelators: D-penicillamine (750-1500 mg/day) or Trientine dihydrochloride (900-1500 mg/day). Maintenance therapy or presymptomatic individuals can be managed with oral zinc acetate (50 mg TID) to block intestinal metallothionein copper absorption.`
  },
  {
    id: "doc-aasld-chelation",
    title: "AASLD Practice Guidance on Wilson Disease Diagnosis & Chelation Protocols",
    filename: "AASLD_Wilson_Guidance_Chelation_2024.pdf",
    category: "Pharmacotherapy",
    format: "PDF",
    fileSize: "3.2 MB",
    uploadDate: "2026-10-28",
    status: "Active",
    chunkCount: 98,
    tokenCount: 49800,
    usageCount: 1420,
    avgLatencyMs: 32,
    similarityThreshold: 0.84,
    accessPermission: "Public CDSS",
    description: "American Association for the Study of Liver Diseases management manual covering penicillamine desensitization and zinc maintenance.",
    content: `Chelation therapy must be initiated with low starting doses to mitigate paradoxical neurologic deterioration. D-penicillamine should begin at 250-500 mg/day, titrating weekly by 250 mg increments to a target of 1000-1500 mg/day divided into 2-4 daily doses, taken 1 hour before or 2 hours after meals. Pyridoxine (vitamin B6) 25-50 mg daily is mandatory co-medication.

Trientine dihydrochloride or tetrahydrochloride is favored in patients exhibiting dermatologic reactions, nephrotoxicity, or early neurologic worsening under penicillamine. Serum free (non-ceruloplasmin-bound) copper should be monitored every 3 months during induction, with a target therapeutic window of 5-15 µg/dL.

Monitoring adherence involves periodic 24-hour urinary copper excretion: stable chelated patients should yield 200-500 µg/day on therapy, whereas zinc-monotherapy patients should display <75 µg/day.`
  },
  {
    id: "doc-atp7b-genetics",
    title: "ATP7B Gene Mutation Spectrum & Genetic Variant Classification Database",
    filename: "ATP7B_Mutation_Database_Consensus.docx",
    category: "Genetics",
    format: "DOCX",
    fileSize: "2.1 MB",
    uploadDate: "2026-11-01",
    status: "Active",
    chunkCount: 64,
    tokenCount: 32600,
    usageCount: 890,
    avgLatencyMs: 29,
    similarityThreshold: 0.88,
    accessPermission: "Restricted Research",
    description: "Exon-by-exon pathogenicity catalog covering p.H1069Q, p.R778L, and splice-site variants.",
    content: `Over 900 pathogenic variants have been recorded across the 21 exons of the ATP7B copper-transporting P-type ATPase gene. In European and North American populations, the missense mutation p.H1069Q in exon 14 accounts for 40-70% of alleles, typically correlating with late-onset neuropsychiatric manifestations.

In East Asian cohorts, p.R778L in exon 8 predominates (30-45% allele frequency), frequently presenting in childhood or early adolescence with acute or progressive hepatic failure. Compound heterozygosity is common.

Next-generation sequencing panels encompassing whole ATP7B coding regions, exon-intron boundaries, and multiplex ligation-dependent probe amplification (MLPA) for gross copy-number deletions yield diagnostic sensitivity exceeding 98%.`
  },
  {
    id: "doc-leipzig-manual",
    title: "Hepatic & Neurologic Leipzig Clinical Decision Matrix Scoring Manual",
    filename: "Leipzig_Score_Clinical_Calculation_Manual.txt",
    category: "Hepatology",
    format: "TXT",
    fileSize: "840 KB",
    uploadDate: "2026-11-04",
    status: "Active",
    chunkCount: 42,
    tokenCount: 21400,
    usageCount: 2150,
    avgLatencyMs: 24,
    similarityThreshold: 0.86,
    accessPermission: "Public CDSS",
    description: "Multi-parameter diagnostic rubric: KF rings, neurological signs, ceruloplasmin, Coombs-negative hemolytic anemia, and hepatic copper.",
    content: `The Leipzig Scoring System (8th International Conference on Wilson Disease) stratifies diagnostic probability based on cumulative clinical, biochemical, and genetic features:

1. Kayser-Fleischer Rings: Present = 2; Absent = 0.
2. Neurologic Symptoms or characteristic brain MRI (copper basal ganglia hyperintensities): Severe = 2; Mild = 1; Absent = 0.
3. Serum Ceruloplasmin: Normal (>0.20 g/L) = 0; Moderately reduced (0.10-0.20 g/L) = 1; Markedly reduced (<0.10 g/L) = 2.
4. Coombs-Negative Hemolytic Anemia: Present = 1; Absent = 0.
5. Liver Copper Content: >250 µg/g dry weight = 2; 50-250 µg/g = 1; Normal (<50 µg/g) = -1.
6. Urinary Copper (absence of acute hepatitis): >2x ULN = 2; 1-2x ULN = 1; Normal = 0.
7. ATP7B Mutation Analysis: Both chromosomes mutant = 4; Single chromosome = 1; None detected = 0.

Interpretation: Total Score >= 4 confirms diagnosis of Wilson's Disease; Score 3 indicates probable disease requiring further workup; Score <= 2 suggests disease is unlikely.`
  },
  {
    id: "doc-pediatric-copper",
    title: "Pediatric vs Adult Copper Kinetics & 24h Urine Copper Benchmarks",
    filename: "Pediatric_Copper_Excretion_Standards.pdf",
    category: "Pediatrics",
    format: "PDF",
    fileSize: "1.9 MB",
    uploadDate: "2026-11-06",
    status: "Active",
    chunkCount: 52,
    tokenCount: 26500,
    usageCount: 670,
    avgLatencyMs: 31,
    similarityThreshold: 0.80,
    accessPermission: "Public CDSS",
    description: "Pediatric specific reference ranges for urinary copper excretion, D-penicillamine challenge testing, and age-stratified ceruloplasmin.",
    content: `Physiologic ceruloplasmin concentrations are naturally diminished in infants under 6 months of age (mean 5-10 mg/dL), reaching adult levels by age 2. Consequently, biochemical screening for Wilson disease before age 2 yields false positives and is unreliable.

For children older than 3 years presenting with unexplained transaminitis, hepatosplenomegaly, or acute Coombs-negative hemolytic anemia, 24-hour urinary copper excretion >40 µg/day (>0.64 µmol/day) warrants high diagnostic suspicion.

The standardized D-penicillamine challenge test (500 mg orally at 0 and 12 hours) measuring 24-hour urinary excretion: values >1600 µg/24h (>25 µmol/day) strongly support diagnosis in pediatric cases with equivocal basal values.`
  },
  {
    id: "doc-dietary-copper",
    title: "Low-Copper Dietary Protocol & Nutritional Restriction Consensus",
    filename: "Dietary_Copper_Restriction_Protocols.docx",
    category: "Pharmacotherapy",
    format: "DOCX",
    fileSize: "1.4 MB",
    uploadDate: "2026-11-09",
    status: "Processing",
    chunkCount: 36,
    tokenCount: 18200,
    usageCount: 340,
    avgLatencyMs: 35,
    similarityThreshold: 0.85,
    accessPermission: "Chief Clinicians Only",
    description: "Nutritional copper avoidance guidelines: shellfish, organ meats, chocolate, nuts, and residential copper plumbing filtration.",
    content: `Dietary copper restriction forms an essential adjunctive therapy during the initial 12-24 months of de-coppering. Clinicians should educate patients on eliminating high-copper food sources:

1. Organ meats (beef liver contains up to 10 mg/100g, sheep liver >15 mg/100g).
2. Shellfish (oysters, crab, lobster, mussels).
3. Nuts and seeds (cashews, walnuts, sunflower seeds, tahini).
4. Dark chocolate and cocoa powders (>0.5 mg/100g).
5. Mushrooms (shiitake, portobello).

Residential drinking water from unlined copper piping must be tested; concentrations should remain <0.1 mg/L. Patients are advised to run taps for 2 minutes prior to collecting cooking or drinking water.`
  }
];

export const INITIAL_QUERY_LOGS = [
  {
    id: "log-101",
    timestamp: "2026-11-12 11:42:15",
    clinician: "Dr. Admin (Department Head)",
    question: "What is the recommended titration protocol for Trientine dihydrochloride in neurological patients?",
    citedDocuments: ["AASLD Practice Guidance on Wilson Disease Diagnosis & Chelation Protocols"],
    citedChunks: ["CHK-AASLD-002", "CHK-AASLD-005"],
    similarityScore: 0.94,
    responseLatencyMs: 138,
    status: "Verified",
    flagged: false,
    feedback: "High Precision"
  },
  {
    id: "log-102",
    timestamp: "2026-11-12 11:28:44",
    clinician: "Dr. Patel (Pediatric Hepatology)",
    question: "How to interpret ceruloplasmin in infants under 12 months with elevated ALT?",
    citedDocuments: ["Pediatric vs Adult Copper Kinetics & 24h Urine Copper Benchmarks"],
    citedChunks: ["CHK-PED-001", "CHK-PED-003"],
    similarityScore: 0.91,
    responseLatencyMs: 146,
    status: "Verified",
    flagged: false,
    feedback: "Clinical Match"
  },
  {
    id: "log-103",
    timestamp: "2026-11-12 10:55:02",
    clinician: "Dr. Chen (Neurology Consultant)",
    question: "Is D-penicillamine safe in patients with pre-existing lupus or proteinuria?",
    citedDocuments: ["AASLD Practice Guidance on Wilson Disease Diagnosis & Chelation Protocols", "EASL Clinical Practice Guidelines: Wilson's Disease (2024 Update)"],
    citedChunks: ["CHK-AASLD-004", "CHK-EASL-012"],
    similarityScore: 0.88,
    responseLatencyMs: 152,
    status: "Verified",
    flagged: false,
    feedback: "Accurate"
  },
  {
    id: "log-104",
    timestamp: "2026-11-12 09:18:30",
    clinician: "Dr. Rivera (Resident Clinician)",
    question: "What Leipzig score threshold indicates mandatory genetic confirmation when liver biopsy is contraindicated?",
    citedDocuments: ["Hepatic & Neurologic Leipzig Clinical Decision Matrix Scoring Manual"],
    citedChunks: ["CHK-LEIP-002"],
    similarityScore: 0.96,
    responseLatencyMs: 119,
    status: "Verified",
    flagged: false,
    feedback: "High Precision"
  },
  {
    id: "log-105",
    timestamp: "2026-11-11 16:40:11",
    clinician: "Dr. Martinez (Clinical Fellow)",
    question: "Are dried fruits and avocados restricted in maintenance copper diet?",
    citedDocuments: ["Low-Copper Dietary Protocol & Nutritional Restriction Consensus"],
    citedChunks: ["CHK-DIET-001"],
    similarityScore: 0.74,
    responseLatencyMs: 178,
    status: "Under Review",
    flagged: true,
    flagReason: "Document status was Processing during query execution; citations need re-verification.",
    feedback: "Needs Guideline Update"
  }
];

export const SYSTEM_KPIS = {
  totalDocuments: 6,
  activeSources: 5,
  avgQueryResponseTimeMs: 142,
  userSatisfactionScore: 98.4,
  totalTokensIndexed: 220800,
  totalVectorDimensions: 3072,
  vectorStoreEngine: "Qdrant HNSW Core",
  embeddingModel: "text-embedding-3-large",
  uptime: "99.98%",
  lastIndexSync: "2 mins ago"
};

export const VECTOR_STORE_CONFIG = {
  modelName: "text-embedding-3-large",
  provider: "OpenAI Clinical Deployment (Azure HIPAA)",
  dimensions: 3072,
  distanceMetric: "Cosine Similarity",
  hnswM: 16,
  hnswEfConstruction: 128,
  hnswEfSearch: 64,
  indexStatus: "Optimal",
  totalVectors: 434,
  memoryUsageMb: 86.4,
  quantization: "Scalar (SQ8) Enabled",
  vectorDbStatus: "Connected (Cluster Health: 100%)"
};

export const EMBEDDING_QUEUE_JOBS = [
  {
    id: "job-embed-901",
    documentTitle: "Dietary_Copper_Restriction_Protocols.docx",
    status: "Processing",
    progress: 68,
    chunksTotal: 36,
    chunksCompleted: 24,
    speedChunksPerSec: 4.8,
    tokensConsumed: 12400,
    startTime: "2 mins ago",
    error: null
  },
  {
    id: "job-embed-900",
    documentTitle: "Pediatric_Copper_Excretion_Standards.pdf",
    status: "Completed",
    progress: 100,
    chunksTotal: 52,
    chunksCompleted: 52,
    speedChunksPerSec: 6.2,
    tokensConsumed: 26500,
    startTime: "18 mins ago",
    error: null
  },
  {
    id: "job-embed-899",
    documentTitle: "Leipzig_Score_Clinical_Calculation_Manual.txt",
    status: "Completed",
    progress: 100,
    chunksTotal: 42,
    chunksCompleted: 42,
    speedChunksPerSec: 7.1,
    tokensConsumed: 21400,
    startTime: "1 hour ago",
    error: null
  },
  {
    id: "job-embed-898",
    documentTitle: "Corrupted_Unstandardized_Genetics_Scan.pdf",
    status: "Failed",
    progress: 14,
    chunksTotal: 48,
    chunksCompleted: 7,
    speedChunksPerSec: 0.0,
    tokensConsumed: 3200,
    startTime: "3 hours ago",
    error: "TokenLimitExceededError: Chunk #8 exceeded max token context window (2450 tokens > 2048 threshold). Embedding model rejected input payload.\n  at VectorTransformer.embedBatch (rag_embedder.py:284)\n  at WorkerPool.processQueue (pipeline_worker.py:112)"
  }
];

/**
 * Generates realistic text chunks and computes overlap boundaries dynamically
 */
export function generateDocumentChunks(doc, method = 'recursive', chunkSize = 512, chunkOverlap = 64) {
  if (!doc || !doc.content) return [];

  const text = doc.content;
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];

  // Approximate tokens: 1 word ~ 1.33 tokens
  const wordsPerChunk = Math.max(15, Math.floor(chunkSize / 1.33));
  const overlapWords = Math.min(Math.floor(wordsPerChunk * 0.45), Math.max(5, Math.floor(chunkOverlap / 1.33)));

  let startIndex = 0;
  let chunkIdx = 1;

  while (startIndex < words.length) {
    const endIndex = Math.min(startIndex + wordsPerChunk, words.length);
    const chunkWords = words.slice(startIndex, endIndex);
    const chunkText = chunkWords.join(' ');
    
    // Determine overlap text with previous chunk
    let overlapPrefix = '';
    if (chunkIdx > 1 && overlapWords > 0) {
      overlapPrefix = words.slice(startIndex, Math.min(startIndex + overlapWords, endIndex)).join(' ');
    }

    const estimatedTokens = Math.round(chunkWords.length * 1.33);

    chunks.push({
      id: `CHK-${doc.id.toUpperCase().replace('DOC-', '')}-${String(chunkIdx).padStart(3, '0')}`,
      index: chunkIdx,
      tokenCount: estimatedTokens,
      charLength: chunkText.length,
      method: method,
      text: chunkText,
      overlapText: overlapPrefix,
      hasOverlap: overlapPrefix.length > 0,
      metadata: {
        docId: doc.id,
        docTitle: doc.title,
        category: doc.category,
        pageRange: `p. ${Math.ceil(chunkIdx / 3)}`,
        startWord: startIndex,
        endWord: endIndex
      }
    });

    if (endIndex >= words.length) break;
    startIndex += (wordsPerChunk - overlapWords);
    chunkIdx++;
  }

  return chunks;
}

/**
 * Simulates real-time vector search across ingested documents
 */
export function simulateVectorSearch(query, topK = 4, similarityThreshold = 0.70, selectedDocId = 'all') {
  const queryNormalized = (query || '').toLowerCase().trim();
  const allDocs = selectedDocId === 'all' 
    ? INITIAL_DOCUMENTS.filter(d => d.status === 'Active')
    : INITIAL_DOCUMENTS.filter(d => d.id === selectedDocId);

  // Measure latency breakdown
  const embeddingTimeMs = Math.floor(12 + Math.random() * 12); // 12-24ms
  const searchTimeMs = Math.floor(16 + Math.random() * 18);    // 16-34ms
  const totalLatencyMs = embeddingTimeMs + searchTimeMs;

  const results = [];

  allDocs.forEach(doc => {
    const chunks = generateDocumentChunks(doc, 'recursive', 512, 64);
    chunks.forEach(chunk => {
      // Calculate realistic cosine similarity score based on term presence and semantics
      const chunkLower = chunk.text.toLowerCase();
      let matchCount = 0;
      const terms = queryNormalized.split(/\s+/).filter(w => w.length > 2);

      terms.forEach(term => {
        if (chunkLower.includes(term)) matchCount++;
      });

      // Synthetic baseline + term relevance
      let score = 0.62 + (matchCount / Math.max(1, terms.length)) * 0.35 + (Math.random() * 0.04 - 0.02);
      score = Math.min(0.98, Math.max(0.45, parseFloat(score.toFixed(3))));

      if (score >= similarityThreshold) {
        results.push({
          chunk,
          cosineScore: score,
          docTitle: doc.title,
          category: doc.category,
          tokenCount: chunk.tokenCount,
          relevanceTier: score >= 0.88 ? 'High Confidence' : (score >= 0.78 ? 'Moderate Match' : 'Borderline Candidate')
        });
      }
    });
  });

  // Sort descending by cosine similarity score
  results.sort((a, b) => b.cosineScore - a.cosineScore);

  return {
    query,
    totalRetrieved: results.length,
    results: results.slice(0, topK),
    latencyBreakdown: {
      embeddingTimeMs,
      searchTimeMs,
      totalLatencyMs
    },
    vectorStoreStats: {
      vectorDimensions: VECTOR_STORE_CONFIG.dimensions,
      distanceMetric: VECTOR_STORE_CONFIG.distanceMetric,
      modelUsed: VECTOR_STORE_CONFIG.modelName
    }
  };
}

