## Week 7: Text and Multimodal AI

**Course:** Introduction to Applied AI (BS 8th Semester, Department of Electronics)
**Project 2:** Deep Learning Applications 
**Domain:** steel surface defect inspection (NEU-DET, six defect classes)
**Notebook:** `Week 7 - Text and Multimodal` (Kaggle, Internet On, GPU optional)

This week my Project 2 system learned to read. I reproduced TF-IDF by hand, compared word and character n-gram classifiers, measured and improved an entity extractor on steel-inspection text, wrote regex fields with negative tests, measured OCR quality with character error rate, and combined my Week 6 image model with text evidence into one masked, explainable JSON record.

### Summary

- Text classifier: SMS Spam Collection (UCI), 5,572 texts, macro-F1 **0.974** (word 1-2 grams) vs **0.982** (char 2-5 grams)
- Top positive / negative features: call, txt, free, claim / my, me, ok, can
- NER on my domain: out-of-the-box held-out F1 **0.667** -> with entity ruler **0.900** (provided sentences: 0.533 -> 0.875)
- Regex fields extracted: CNIC, mobile, amount, date, email, invoice, coil, batch, defect; false positives tested: 5 negative tests, all passed (9 of 9 tests)
- OCR: CER raw 0.062 / 0.059 / 0.143 -> **0.000** after correction rules (synthetic invoice: 0.000 before and after)
- Pipeline: image model + text evidence -> JSON record (example below)
- Privacy: CNIC, phone, and email masked before logging

### Results by part

| Part | What I did | Measured result |
|---|---|---|
| 1. TF-IDF by hand | Reproduced scikit-learn's smooth IDF with L2 normalisation on three messages | d1 = win 0.584, free 0.445, cash 0.584, now 0.345 (identical in both printouts); cosine(d2, d3) = 0.544 |
| 2. Text classification | TF-IDF + Logistic Regression on SMS spam, 80/20 stratified split (1,115 test messages) | See table below |
| 3. NER, measured | Gold set scored by exact text AND label; entity ruler added before the statistical model | Provided set F1 0.533 -> 0.875; my held-out F1 0.667 -> 0.900 |
| 4. Regex fields | Six provided patterns plus three for steel inspection | 9 of 9 tests passed, including 5 negative tests |
| 5. CER and OCR repair | Levenshtein-based CER, context rules, EasyOCR on a synthetic invoice | Hand-made errors 0.062 / 0.059 / 0.143 -> 0.000; invoice 0.000 -> 0.000 |
| 6. Image + text record | Late fusion with transparent rules and PII masking | REFER with two reasons; phone digits leaked: False |

### Text classification (Part 2)

Dataset: 5,572 messages, 4,825 ham and 747 spam (13.4% spam). An "always ham" classifier would score 0.866 accuracy, so accuracy alone is misleading.

| Model | Macro-F1 | Spam precision | Spam recall | Features |
|---|---|---|---|---|
| Word 1-2 grams | 0.974 | 0.979 | 0.933 | 12,128 |
| Char 2-5 grams | **0.982** | **0.986** | **0.953** | 36,449 |

- **Top features (word model).** Most spam-like: call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your. Most ham-like: my, me, ok, that, it, but, ll, gt, lt, can, what, da. Most make sense, but "to" and "your" are ordinary function words, and "gt" and "lt" are leftovers of ">" and "<" encoded as HTML in this dataset, so they are shortcuts rather than meaning.
- **Error analysis (word model, 13 of 1,115).** 10 missed spam and 3 false alarms. I classified them as 7 model errors, 2 label errors (a tattoo joke and a dartboard sale ad labelled spam) and 4 genuinely ambiguous messages. "Waiting for your call." scored 0.85 because "call" is a strong spam feature.
- **Spelling tricks.** With the character model, "W1N a FR33 pr1ze!! txt CLAIM now" scored 0.85 (spam) and the Roman Urdu "kal milte hain, 5 baje" scored 0.12 (ham), both correct. My own tricky messages scored: Roman Urdu spam, word 0.60 and char 0.66; disguised English spam, word 0.86 and char 0.90. Both models caught them, but the Roman Urdu one with much lower confidence.

### Named entity recognition (Part 3)

Entities count as correct only if both the exact text and the label match.

| Set | Before rules (F1) | After rules (F1) |
|---|---|---|
| Provided 6 sentences (16 gold entities) | 0.533 (P 0.571, R 0.500) | 0.875 (P 0.875, R 0.875) |
| My first 5 sentences (rules written from these) | 0.485 | 0.941 |
| **My last 5 sentences (held out)** | **0.667** (P 0.684, R 0.650) | **0.900** (P 0.900, R 0.900) |

The held-out score is the honest one, because the rules were written only from the first five sentences.

- **What spaCy got wrong out of the box:** company names read as PERSON, dd/mm/yyyy dates labelled GPE, PKR and Rs. amounts split or mislabelled, and Pakistani cities labelled ORG or PERSON.
- **Errors that remain after rules:** coil and batch numbers (`4417`, `2205`) labelled DATE, and names the rules had never seen ("Ittefaq Foundry" and "Sialkot").
- **What I would do next:** a regex or a filter that checks the preceding word ("coil", "batch") for the number problem, and a model trained on a few hundred annotated reports for unseen names. Ten sentences are too few to train one.

### Regex fields (Part 4)

Patterns: CNIC, mobile, amount, date, email and invoice number (provided), plus three for steel inspection:

- `coil`: `\bcoil\s+(\d{3,6})\b`
- `batch`: `\bbatch\s+(\d{3,6})\b`
- `defect`: `\b(crazing|inclusion|patches|pitted surface|rolled-in scale|scratches)\b`

All 9 tests passed. The negative tests were: a 14-digit order number (not a CNIC), "Room 0300" (not a mobile), "Coil 12" (too few digits), "coil 4417896" (too many digits) and "A batch of 25 sheets" (no batch number).

My first coil pattern, `coil\s+\d+`, matched all three of my test strings, including the two that must be rejected. A word boundary and an exact digit count (`{3,6}`) fixed it.

### OCR quality (Part 5)

| Reference | OCR output | CER before | CER after |
|---|---|---|---|
| Invoice No. 4417 | Invoice N0. 4417 | 0.062 | 0.000 |
| Total: PKR 12,500 | Total: PKR 12,5O0 | 0.059 | 0.000 |
| Due 30/09/2026 | Due 3O/09/2O26 | 0.143 | 0.000 |

On the synthetic invoice, EasyOCR read all five lines correctly (CER 0.000 before and after), so the correction rules had nothing to repair. That image is black text on white in one clean font, which is much easier than a real phone photo. On real documents I expect a clearly non-zero CER concentrated in digits and look-alike characters, so field-level accuracy should be measured too.

### Image + text case record (Part 6)

Late fusion: my Week 6 ResNet-18 (six classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) gives image probabilities, and the note gives entities, regex fields and urgent terms. Transparent rules combine them. The class I must not miss is **crazing**, since fine cracks can spread.

The note is masked **before** any extraction, so the phone number cannot reappear in the entities or fields.

```json
{
  "image": "crazing_241.jpg",
  "image_probabilities": {
    "crazing": 1.0, "inclusion": 0.0, "patches": 0.0,
    "pitted_surface": 0.0, "rolled-in_scale": 0.0, "scratches": 0.0
  },
  "note_masked": "Inspector Imran Sheikh at Pakistan Steel Mills, Karachi found fine cracks spreading on coil 4417 on 12/09/2026. Contact mobile [PHONE].",
  "text_evidence": {
    "entities": [["Imran Sheikh", "PERSON"], ["Pakistan Steel Mills", "ORG"],
                 ["Karachi", "GPE"], ["4417", "DATE"], ["12/09/2026", "DATE"]],
    "fields": {"date": ["12/09/2026"], "coil": ["4417"]},
    "urgent_terms": ["crack", "spreading"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(crazing) = 1.00", "note mentions: crack, spreading"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

- **A wrong entity:** "4417" is a coil number but was labelled DATE. It is harmless for this decision, which came from the image probability and urgent terms, but harmful if a database stored the entity list. The regex field `coil: ["4417"]` has the right value.
- **When image and note disagree:** with a patches image and the same note, P(crazing) = 0.0 and the decision was REFER on the note alone. With the crazing image and a note saying "Surface looks normal", P(crazing) = 1.0 and the decision was REFER on the image alone. The system should refer such cases and state the disagreement explicitly, which is a possible next improvement.

### Privacy

CNIC numbers, mobile numbers and email addresses are masked (`[CNIC]`, `[PHONE]`, `[EMAIL]`) before extraction and logging. A leak check on the final JSON for the phone digits printed `False`.

### Limitations

- The domain gold set has only 10 sentences, so the NER scores have a wide margin of error.
- The entity rules fix a closed list of names and will not generalise to new ones.
- The image evidence comes from one validation image, so its probability of 1.00 is an example and not a performance claim.
- The OCR test used a clean synthetic image, not real scans.
- The record is decision support only; a qualified human decides.

### How to reproduce

1. Create a Kaggle notebook with **Internet On** (EasyOCR downloads its models on first use).
2. Add the inputs: the *SMS Spam Collection Dataset* (uciml), the *NEU Surface Defect Database*, and my Week 6 notebook output containing `week6_model.pt`.
3. Run all cells from the top.
