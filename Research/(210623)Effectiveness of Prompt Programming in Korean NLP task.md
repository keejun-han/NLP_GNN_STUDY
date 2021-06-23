## Effectiveness of Prompt Programming in Korean NLP task

### Related Works
- Automatic search for discrete prompt
  - (2019/11/28) `How can we know what language models know?`
  - (2021/02/15) `Prompt Programming for Large Language Models: Beyond the Few-Show Paradigm`
- Prefix-tuning
  - (2021/01/01) `Prefix-Tuning: Optimizing coninous prompts for generation`
  - Was mentioned in `Naver AI NOW` (Data augmentation part)
- P-tuning
  - (2021/03/18) `GPT Understands, Too`
  - Was mentioned in `Naver AI NOW` (Data augmentation part)
- Prompt-Tuning
  - (2021/04/18) `The Power of Scale for Parameter-Efficient Prompt Tuning`

### Method
- Compare performance of **Prefix-tuning, P-tuning, Prompt-tuning, and Fine-tuning** in Korean Downstream Task
  - Have not decided which downstream task to apply
  - Can compare in more than one downstream task
- Apply additional method focusing to increase performance in Korean task
  - e.g. Considering grammatical patterns, slangs for informal spoken langauge

### Experiment
- Can apply in Korean GPT style model / KoBERT
  - For GPT style, can refer to [SKT AI's KoGPT2](https://github.com/SKT-AI/KoGPT2)
  - For BERT style, can refer to [SKT Brain's KoBERT](https://github.com/SKTBrain/KoBERT)

### Contribution
- As 'prompt programming' is relatively new, there would be less studies on methodolgy's comparison
- We can find which method is effective or efficient in each downstream task / specific downstream task
  - Can compare accuarcy / consumed time / human effort
- For each result, we can suggest our novel reason (interpretation)
