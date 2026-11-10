# Lesson 1: why the participant is the unit of evaluation

An MRI volume can produce many related slices. If one person's slices enter both training and testing, the model may recognize person-specific features rather than generalize to a new person. We therefore assign a split to each participant and then propagate it to every image.

Read `partition()` in this order:

1. Required-field checks prevent incomplete records from entering experiments.
2. The participant dictionary checks that labels are consistent.
3. Grouping by label maintains class representation across splits.
4. Sorting and a local seeded random generator make assignment repeatable, even if CSV row order changes.
5. New dictionaries preserve the original input rows.

This checks identifiers, not biological identity. Incorrect source IDs, duplicate images under different paths, longitudinal targets, and demographic confounding require additional audits.

Exercises:

1. Explain why 100 slices from one participant do not equal 100 independent participants.
2. Run the synthetic example; predict participant counts before reading the output.
3. Change the seed and write down what changes and what must stay true.
4. Introduce conflicting labels and explain the resulting error.
5. Explain why a consistent split alone does not prevent preprocessing leakage.

Daily study suggestion: 45 minutes reading, 90 minutes running/modifying code, 45 minutes writing observations and questions. Keep your answers locally until you are comfortable publishing them.
