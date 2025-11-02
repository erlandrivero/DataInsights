# 🚀 Quick Setup: Google AI API Key

## Step 1: Get API Key (2 minutes)

1. **Go to:** https://aistudio.google.com/app/apikey
2. **Sign in** with Google account
3. **Click:** "Create API Key"
4. **Copy** the key (starts with `AIza...`)

---

## Step 2: Add to Local Environment (1 minute)

**Edit `.env` file in project root:**

```env
GOOGLE_API_KEY=AIza...paste-your-key-here...
```

💡 **Tip:** If `.env` doesn't exist, copy from `.env.example`

---

## Step 3: Add to Streamlit Cloud (Optional - 2 minutes)

1. **Go to:** Your app on Streamlit Cloud
2. **Click:** ⚙️ Settings → 🔐 Secrets
3. **Add:**
   ```toml
   GOOGLE_API_KEY = "AIza...your-key..."
   ```
4. **Click:** Save

---

## ✅ Done!

Your configuration is ready. The key is now available for the migration.

**Next:** See `AI_PROVIDER_MIGRATION_GUIDE.md` for full migration steps.

---

## 🔗 Quick Links

- **Get API Key:** https://aistudio.google.com/app/apikey
- **Manage Keys:** https://aistudio.google.com/
- **Documentation:** https://ai.google.dev/docs
- **Pricing:** https://ai.google.dev/pricing (Free tier: 60 req/min)

---

## 💰 Cost Comparison

| Feature | OpenAI GPT-4 | Google Gemini |
|---------|--------------|---------------|
| Input (1M tokens) | $10 | $1.25 (**8x cheaper**) |
| Output (1M tokens) | $30 | $5 (**6x cheaper**) |
| Free tier | ❌ None | ✅ 60 req/min |
| Context window | 128K | 1M (**8x larger**) |

**Your estimated savings: 85%** 💸
