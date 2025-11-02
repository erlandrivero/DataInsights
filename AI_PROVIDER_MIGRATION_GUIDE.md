# AI Provider Migration Guide
## Switching from OpenAI to Google AI (Gemini)

**Last Updated:** November 2, 2025  
**Status:** Configuration Ready ✅

---

## 📋 **Quick Start: Adding Your Google API Key**

### **Step 1: Get Your Google API Key**

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

---

### **Step 2A: Local Development Setup**

**Edit your `.env` file** (create if doesn't exist):

```env
# OpenAI Configuration (current)
OPENAI_API_KEY=sk-proj-...your-actual-key...

# Google AI Configuration (new - add your key here)
GOOGLE_API_KEY=AIza...your-actual-google-key...

# Active Provider (when ready to switch, uncomment google)
# AI_PROVIDER=openai
# AI_PROVIDER=google
```

---

### **Step 2B: Streamlit Cloud Setup**

1. Go to your app on Streamlit Cloud
2. Click **Settings** → **Secrets**
3. Add to secrets:

```toml
# OpenAI Configuration (current)
OPENAI_API_KEY = "sk-proj-...your-actual-key..."

# Google AI Configuration (new - add your key here)
GOOGLE_API_KEY = "AIza...your-actual-google-key..."

# Active Provider
AI_PROVIDER = "openai"
```

4. Click **Save**

---

## 🔄 **Migration Checklist**

### **Phase 1: Preparation** ✅ COMPLETE

- [x] Updated `.env.example` with Google API key section
- [x] Updated `.streamlit/secrets.toml.example` with Google configuration
- [x] Created migration guide (this file)
- [ ] **ACTION REQUIRED:** Add your Google API key to `.env`
- [ ] **ACTION REQUIRED:** Add your Google API key to Streamlit Cloud secrets

### **Phase 2: Code Migration** (Ready to implement)

- [ ] Update `requirements.txt` (add `google-generativeai`, remove `openai`)
- [ ] Rewrite `utils/ai_helper.py` for Gemini
- [ ] Update `utils/ai_smart_detection.py` API calls
- [ ] Update 14 AI insight calls in `app.py`
- [ ] Test all modules with Gemini
- [ ] Update error handling

### **Phase 3: Deployment** (After testing)

- [ ] Push code changes to GitHub
- [ ] Update Streamlit Cloud secrets to use `AI_PROVIDER = "google"`
- [ ] Verify all modules work on production
- [ ] Remove OpenAI dependency

---

## 💡 **Benefits of Switching to Google AI**

| Metric | OpenAI GPT-4 | Google Gemini 1.5 Pro | Improvement |
|--------|--------------|----------------------|-------------|
| **Cost per 1M input tokens** | $10.00 | $1.25 | **8x cheaper** |
| **Cost per 1M output tokens** | $30.00 | $5.00 | **6x cheaper** |
| **Context window** | 128K tokens | 1M tokens | **8x larger** |
| **Rate limits (free tier)** | None | 60 req/min | **Free usage!** |
| **Your estimated monthly cost** | ~$15-20 | ~$2-3 | **85% savings** |

---

## 📝 **Files That Will Change**

### **Configuration Files** (Update manually)
1. `.env` - Add `GOOGLE_API_KEY` (local development)
2. Streamlit Cloud Secrets - Add `GOOGLE_API_KEY` (production)

### **Code Files** (Will be modified during migration)
1. `requirements.txt` - Swap openai → google-generativeai
2. `utils/ai_helper.py` - Complete rewrite for Gemini API
3. `utils/ai_smart_detection.py` - Update API calls (~line 589)
4. `app.py` - Update 14 AI insight generation calls

---

## 🚀 **Next Steps**

### **Immediate Actions:**

1. **Get your Google API key**: https://aistudio.google.com/app/apikey

2. **Add to local `.env`** (copy from `.env.example` if needed):
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env and add your keys
   notepad .env
   ```

3. **Test key validity** (optional - run in Python):
   ```python
   import google.generativeai as genai
   genai.configure(api_key="AIza...your-key...")
   model = genai.GenerativeModel('gemini-1.5-pro')
   response = model.generate_content("Hello! Test message.")
   print(response.text)
   ```

4. **Add to Streamlit Cloud secrets** (if deploying):
   - App Settings → Secrets
   - Add `GOOGLE_API_KEY = "your-key"`

---

## ⚠️ **Important Notes**

### **API Key Security**
- ✅ `.env` is in `.gitignore` (never commits)
- ✅ `secrets.toml` is in `.gitignore` (never commits)
- ✅ Only `.example` files are tracked in git
- ⚠️ **Never share your API keys publicly**

### **Cost Management**
- **Google AI Free Tier**: 60 requests/minute (sufficient for most use)
- **Paid tier**: Only charged for what you use
- **Monitor usage**: https://aistudio.google.com/app/apikey

### **Testing Strategy**
- Test locally first with sample data
- Verify all 14 modules generate AI insights
- Compare output quality with OpenAI
- Check JSON parsing (Gemini requires explicit formatting)

---

## 📞 **Support Resources**

### **Google AI Studio**
- Dashboard: https://aistudio.google.com/
- Documentation: https://ai.google.dev/docs
- Python SDK: https://ai.google.dev/tutorials/python_quickstart

### **API Keys**
- Create/Manage: https://aistudio.google.com/app/apikey
- Pricing: https://ai.google.dev/pricing

---

## 🎯 **Migration Timeline**

**Estimated Time:** 2-3 hours

1. **Configuration (5 min)** - Add API key ← **YOU ARE HERE**
2. **Code Changes (1-2 hours)** - Migrate utilities and app.py
3. **Testing (30 min)** - Verify all modules
4. **Deployment (15 min)** - Push to production
5. **Validation (15 min)** - Confirm everything works

---

## ✅ **Ready to Migrate?**

When you're ready to proceed with the code migration, we'll:

1. Create `utils/ai_helper_gemini.py` (new Gemini-compatible version)
2. Create migration script for `app.py` updates
3. Update `requirements.txt`
4. Test all 14 modules
5. Deploy to production

**Let me know when you want to start Phase 2!** 🚀
