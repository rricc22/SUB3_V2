# Hugging Face Deployment Package

This folder contains all files needed to deploy your Heart Rate Prediction model on Hugging Face.

## Files Included (Total: ~17 MB)

### Core Application Files
- **app.py** (23 KB) - Streamlit application with UI
- **best_model.pt** (2.4 MB) - Trained LSTM model checkpoint
- **requirements_hf.txt** (77 B) - Python dependencies
- **Dockerfile** (778 B) - Docker configuration for HF Spaces
- **README.md** (1.3 KB) - Space description (appears on HF Space page)

### Animation Files (14 MB total)
- **steady_workouts.gif** (5.3 MB) - Steady pace run examples
- **intervals_workouts.gif** (4.5 MB) - Interval training examples
- **progressive_workouts.gif** (1.7 MB) - Progressive run examples
- **category_comparison.gif** (2.5 MB) - Side-by-side comparison

### Sample Data Files (52 KB total)
- **sample_steady.json** (14 KB) - Real workout data for steady pace
- **sample_intervals.json** (19 KB) - Real workout data for intervals
- **sample_progressive.json** (20 KB) - Real workout data for progressive runs

---

## Deployment Instructions

### Option 1: Deploy via Hugging Face CLI

```bash
# Activate your conda environment
conda activate ai_generali

# Navigate to deployment folder
cd /home/riccardo/Documents/SUB3_V2/huggingface_deployment

# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Login to Hugging Face (you'll need a write-access token)
huggingface-cli login

# Create a new Space (first time only)
huggingface-cli repo create heart-rate-predictor --type space --space_sdk docker

# Upload all files to your Space
huggingface-cli upload rricc22/heart-rate-predictor . --repo-type space
```

### Option 2: Deploy via Web Interface

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name it: `heart-rate-predictor`
4. Select SDK: **Docker**
5. Upload all files from this folder
6. Wait for the Space to build (~2-3 minutes)

### Option 3: Deploy via Git (Advanced)

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/rricc22/heart-rate-predictor
cd heart-rate-predictor

# Copy all files from this folder
cp /home/riccardo/Documents/SUB3_V2/huggingface_deployment/* .

# Commit and push
git add .
git commit -m "Deploy heart rate prediction model"
git push
```

---

## File Size Optimization (Optional)

If you want to reduce deployment size, you can:

1. **Remove GIF animations** (~14 MB savings)
   - The app has fallback logic to show placeholders
   - Delete: `*.gif` files
   
2. **Use smaller model checkpoint** (if available)
   - Current: 2.4 MB
   - Consider model quantization or pruning

3. **Compress GIF files** (keep animations but reduce size)
   ```bash
   # Install gifsicle
   sudo apt-get install gifsicle
   
   # Compress GIFs (lossy compression)
   gifsicle --optimize=3 --lossy=80 -o steady_workouts_small.gif steady_workouts.gif
   ```

---

## Testing Your Deployment

Once deployed, your Space will be available at:
**https://huggingface.co/spaces/rricc22/heart-rate-predictor**

Test these features:
- [ ] Sample Workouts tab loads with animations
- [ ] Clicking "Predict" buttons generates results
- [ ] Manual Input tab accepts comma-separated values
- [ ] CSV Upload tab processes uploaded files
- [ ] Download buttons provide CSV results
- [ ] Plots display correctly

---

## Troubleshooting

### Build fails with memory error
- Remove GIF animations to reduce size
- Use smaller Docker base image

### App crashes on startup
- Check logs in HF Space "Build" tab
- Verify `best_model.pt` is not corrupted
- Ensure `requirements_hf.txt` has correct versions

### Predictions are wrong
- Verify model checkpoint is the correct version
- Check feature engineering matches training code
- Ensure normalization is consistent

---

## Updating Your Deployment

To update the Space with new changes:

```bash
cd /home/riccardo/Documents/SUB3_V2/huggingface_deployment

# Make your changes to app.py or other files
# Then re-upload
huggingface-cli upload rricc22/heart-rate-predictor . --repo-type space
```

---

## Current Deployment Status

According to `DEPLOYMENT_SUMMARY.md`, your model is already deployed:
- **Model Hub**: https://huggingface.co/rricc22/heart-rate-prediction-lstm
- **Interactive Demo**: https://huggingface.co/spaces/rricc22/heart-rate-predictor

This folder contains a clean copy of all files for future deployments or updates.

---

**Last Updated**: January 14, 2026  
**Total Package Size**: ~17 MB
