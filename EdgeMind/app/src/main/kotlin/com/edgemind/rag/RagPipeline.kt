package com.edgemind.rag

import android.content.Context
import android.util.Log
import com.edgemind.data.RagChunk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * On-device RAG pipeline.
 * Embeds the query, retrieves top-k chunks from the FAISS index,
 * and returns the augmented context string for re-inference.
 */
class RagPipeline(private val context: Context) {

    private val embedModel = EmbedModel(context)
    private val index = FaissIndex(EmbedModel(context).dimension)
    private val chunks = mutableMapOf<Long, RagChunk>()
    private var indexReady = false

    suspend fun initialize() = withContext(Dispatchers.IO) {
        embedModel.load()
        loadOrBuildIndex()
        indexReady = true
        Log.i(TAG, "RAG pipeline ready. Chunks: ${chunks.size}")
    }

    suspend fun retrieve(query: String, k: Int = 5): List<RagChunk> =
        withContext(Dispatchers.Default) {
            if (!indexReady) return@withContext emptyList()

            val start = System.currentTimeMillis()
            val queryEmbedding = embedModel.embed(query)
            val results = index.search(queryEmbedding, k)
            Log.d(TAG, "FAISS search: ${System.currentTimeMillis() - start} ms")

            results.mapNotNull { (id, score) ->
                chunks[id]?.copy(score = score)
            }
        }

    fun buildAugmentedPrompt(originalPrompt: String, chunks: List<RagChunk>): String {
        val context = chunks.joinToString("\n---\n") { it.text }
        return """Clinical Knowledge Context:
$context

---

Patient Query: $originalPrompt

Based on the clinical context above, provide a structured triage assessment."""
    }

    private suspend fun loadOrBuildIndex() {
        val indexFile = context.filesDir.resolve("clinical_kb.faiss")
        if (indexFile.exists()) {
            index.loadFrom(indexFile.absolutePath)
            loadChunkMetadata()
        } else {
            buildDemoIndex()
            index.save(indexFile.absolutePath)
        }
    }

    private fun loadChunkMetadata() {
        val metaFile = context.filesDir.resolve("kb_meta.txt")
        if (!metaFile.exists()) {
            buildDemoIndex()
            return
        }
        metaFile.forEachLine { line ->
            val parts = line.split("|", limit = 2)
            if (parts.size == 2) {
                val id = parts[0].toLongOrNull() ?: return@forEachLine
                chunks[id] = RagChunk(id = id, text = parts[1], score = 0f)
            }
        }
    }

    private fun buildDemoIndex() {
        DEMO_KNOWLEDGE_BASE.forEachIndexed { idx, text ->
            val id = idx.toLong()
            val chunk = RagChunk(id = id, text = text, score = 0f)
            chunks[id] = chunk
            // Build embedding synchronously for index construction
            val embedding = buildDeterministicEmbedding(text, embedModel.dimension)
            index.add(embedding, chunk)
        }
        persistChunkMetadata()
        Log.i(TAG, "Demo index built with ${chunks.size} chunks")
    }

    private fun buildDeterministicEmbedding(text: String, dim: Int): FloatArray {
        val seed = text.hashCode()
        val rand = java.util.Random(seed.toLong())
        val v = FloatArray(dim) { rand.nextGaussian().toFloat() }
        val norm = Math.sqrt(v.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
        return FloatArray(dim) { v[it] / norm }
    }

    private fun persistChunkMetadata() {
        val metaFile = context.filesDir.resolve("kb_meta.txt")
        metaFile.bufferedWriter().use { writer ->
            chunks.forEach { (id, chunk) ->
                writer.write("$id|${chunk.text.replace("\n", " ")}\n")
            }
        }
    }

    fun close() = embedModel.close()

    companion object {
        private const val TAG = "RagPipeline"

        val DEMO_KNOWLEDGE_BASE = listOf(
            // ESI triage criteria
            "ESI Level 1 — Immediate: Patient requires immediate life-saving intervention. Conditions include cardiac arrest, respiratory failure, severe haemorrhage, altered consciousness (GCS < 9), active seizures.",
            "ESI Level 2 — Emergent: Patient is in a high-risk situation or has acute changes to vital signs that could deteriorate rapidly. Includes chest pain with cardiac risk factors, severe dyspnoea, altered mental status, high-risk mechanism of injury.",
            "ESI Level 2 Criteria: Severe pain (8-10/10), active haemorrhage, acute neurological deficit, severe dehydration, ECG changes suggestive of ischaemia, oxygen saturation < 90% on room air.",
            "ESI Level 3 — Urgent: Stable vital signs but requires 2+ resources. Examples: abdominal pain, complex lacerations, moderate pain, urinary symptoms with fever.",
            "ESI Level 4 — Less Urgent: One resource needed, stable vitals. Examples: minor lacerations, sprains, UTI without systemic signs.",
            "ESI Level 5 — Non-urgent: No resources needed. Well-appearing, stable, minor complaints.",

            // ACS red flags
            "Acute Coronary Syndrome (ACS) Red Flags: Central crushing chest pain radiating to arm/jaw/back, diaphoresis, nausea/vomiting, dyspnoea, syncope or near-syncope. Risk factors: age > 45 (men) or > 55 (women), hypertension, diabetes mellitus, hyperlipidaemia, smoking, family history of premature CAD.",
            "ACS ECG Changes: ST elevation (STEMI) ≥ 1 mm in ≥ 2 contiguous leads, ST depression, new LBBB, T-wave inversions. Posterior MI: ST depression V1-V3 with tall R waves.",
            "Troponin Interpretation: High-sensitivity troponin I or T. Elevated if above 99th percentile URL. Serial measurement at 0 h and 1-3 h. Rising or falling pattern confirms myocardial injury.",
            "MONA Protocol for ACS: Morphine IV (2.5-5 mg, titrate) for pain. Oxygen if SpO2 < 94%. Nitrates (GTN SL or IV) if systolic BP > 90 mmHg. Aspirin 300 mg PO loading dose stat.",

            // Pulmonary embolism
            "Pulmonary Embolism Signs: Acute dyspnoea, pleuritic chest pain, haemoptysis, tachycardia (HR > 100), tachypnoea (RR > 20), hypoxia. Massive PE: haemodynamic instability, shock index > 1.",
            "Wells PE Score: Clinical signs of DVT (+3), PE most likely diagnosis (+3), HR > 100 (+1.5), immobilisation ≥ 3 days or surgery in 4 weeks (+1.5), previous DVT/PE (+1.5), haemoptysis (+1), malignancy (+1). Score ≤ 4: low probability; > 4: high probability.",
            "PE Investigation: D-dimer (sensitive, not specific). CTPA gold standard. V/Q scan if contrast allergy or renal impairment. Echo: RV strain, McConnell sign in massive PE.",

            // Drug interactions
            "Critical Drug Interaction — Warfarin: NSAIDs increase bleeding risk. Antibiotics (especially metronidazole, fluoroquinolones) potentiate anticoagulant effect — INR monitoring required. Avoid aspirin unless clearly indicated.",
            "Critical Drug Interaction — Statins: Avoid simvastatin > 40 mg with amiodarone, amlodipine. Rosuvastatin dose reduction with cyclosporin. Myopathy risk increased with fibrates.",
            "High-alert Medications: Insulin, anticoagulants (warfarin, heparin, DOACs), concentrated electrolytes (KCl), opioids, neuromuscular blocking agents. Require double-check before administration.",

            // Medication normalisation
            "Aspirin: Acetylsalicylic acid. Standard doses: 75-100 mg OD (antiplatelet), 300-600 mg loading (ACS), 600 mg (analgesia). Avoid in children < 16 (Reye syndrome risk).",
            "Metoprolol: Beta-1 selective blocker. Metoprolol succinate (XL) — once daily. Metoprolol tartrate (IR) — BD-TDS. Starting dose: 25 mg BD. Max: 400 mg/day.",
            "Enoxaparin: LMWH. Treatment dose: 1 mg/kg SC BD or 1.5 mg/kg SC OD. Prophylaxis: 40 mg SC OD. Renal impairment (CrCl < 30): reduce dose, consider UFH.",

            // Respiratory
            "Differential Diagnosis Dyspnoea: Cardiac (ACS, heart failure, arrhythmia), Pulmonary (PE, pneumonia, pneumothorax, asthma, COPD exacerbation, pulmonary hypertension), Metabolic (DKA, anaemia, thyrotoxicosis), Neuromuscular (Guillain-Barré, myasthenia gravis).",
            "Tension Pneumothorax: Life-threatening emergency. Clinical diagnosis — do not wait for CXR. Signs: absent breath sounds, tracheal deviation away from affected side, hypotension, JVD, respiratory distress. Treatment: immediate needle decompression (2nd ICS, MCL), followed by chest drain.",

            // Vital signs interpretation
            "SIRS Criteria: Temperature > 38°C or < 36°C, HR > 90, RR > 20 or PaCO2 < 32 mmHg, WBC > 12,000 or < 4,000 or > 10% bands. ≥ 2 criteria = SIRS. SIRS + suspected infection = Sepsis.",
            "Shock Index: HR / Systolic BP. Normal < 0.7. > 1.0 suggests significant haemodynamic compromise. > 1.4 critical. Use to rapidly identify patients requiring immediate resuscitation.",
        )
    }
}
