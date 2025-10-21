# DataInsight AI - Deployment Guide

## Deploying to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- OpenAI API key

### Step-by-Step Deployment

#### 1. Prepare Your Repository

1. **Commit all changes:**
```bash
git add .
git commit -m "Prepare for deployment"
```

2. **Push to GitHub:**
```bash
git push origin main
```

3. **Verify files:**
- [ ] `app.py` exists
- [ ] `requirements.txt` is complete
- [ ] `.streamlit/config.toml` exists
- [ ] `.gitignore` includes `secrets.toml`

#### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Configure:
   - **Branch:** main
   - **Main file path:** app.py
   - **App URL:** your-app-name (choose a unique name)
5. Click "Deploy"

#### 3. Configure Secrets

1. While app is deploying, click "Advanced settings"
2. Go to "Secrets" section
3. Add your secrets:
```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```
4. Click "Save"

#### 4. Wait for Deployment

- Initial deployment takes 2-5 minutes
- Watch the logs for any errors
- App will automatically restart after deployment

#### 5. Test Your Deployed App

1. Visit your app URL: `https://your-app-name.streamlit.app`
2. Test all features:
   - [ ] Data upload works
   - [ ] AI insights generate
   - [ ] Visualizations display
   - [ ] Reports generate
   - [ ] Export functions work

### Troubleshooting

#### App Won't Start

**Problem:** App shows error on startup

**Solutions:**
1. Check requirements.txt has all dependencies
2. Verify Python version compatibility
3. Check logs for specific error messages

#### AI Features Don't Work

**Problem:** AI insights fail to generate

**Solutions:**
1. Verify OpenAI API key is set in secrets
2. Check API key is valid and has credits
3. Verify internet connectivity from Streamlit Cloud

#### Slow Performance

**Problem:** App is slow or times out

**Solutions:**
1. Optimize data processing for large files
2. Add caching with `@st.cache_data`
3. Consider upgrading Streamlit Cloud plan

### Updating Your Deployed App

1. Make changes locally
2. Test thoroughly
3. Commit and push to GitHub:
```bash
git add .
git commit -m "Description of changes"
git push origin main
```
4. Streamlit Cloud will automatically redeploy

### Custom Domain (Optional)

1. In Streamlit Cloud dashboard, go to app settings
2. Click "Custom domain"
3. Follow instructions to set up your domain
4. Update DNS records as instructed

### Monitoring

- Check app logs in Streamlit Cloud dashboard
- Monitor usage statistics
- Set up alerts for errors

### Security Best Practices

1. **Never commit secrets:**
   - Always use `.gitignore` for `secrets.toml`
   - Use environment variables for sensitive data

2. **API Key Security:**
   - Rotate API keys regularly
   - Use separate keys for dev/prod
   - Monitor API usage

3. **Data Privacy:**
   - Don't log sensitive user data
   - Clear session state appropriately
   - Consider data retention policies

### Cost Considerations

**Streamlit Cloud:**
- Free tier: 1 app, limited resources
- Paid tiers: More apps, more resources

**OpenAI API:**
- Pay per token used
- Monitor usage to control costs
- Set usage limits in OpenAI dashboard

### Support

If you encounter issues:
1. Check Streamlit Cloud documentation
2. Visit Streamlit Community Forum
3. Open issue on GitHub repository

---

**Congratulations! Your DataInsight AI app is now deployed! 🎉**
