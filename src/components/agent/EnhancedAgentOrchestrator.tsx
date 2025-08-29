import React, { useRef, useEffect, useCallback, useState } from "react";
import { usePublicClient } from "wagmi";
import { storyAeneid } from "@/lib/chains/story";
import { waitForTxConfirmation } from "@/lib/utils/transaction";
import { useChatAgent } from "@/hooks/useChatAgent";
import { useSwapAgent } from "@/hooks/useSwapAgent";
import { useRegisterIPAgent } from "@/hooks/useRegisterIPAgent";
import { useFileUpload } from "@/hooks/useFileUpload";
import { DEFAULT_LICENSE_SETTINGS } from "@/lib/license/terms";
import type { LicenseSettings } from "@/lib/license/terms";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { PlanBox } from "./PlanBox";
import { HistorySidebar } from "./HistorySidebar";
import { Toast } from "./Toast";
import { CameraCapture } from "./CameraCapture";
import { detectAI, fileToBuffer, detectIPStatus } from "@/services";
import { isWhitelistedImage, computeDHash } from "@/lib/utils/whitelist";
import { compressImage } from "@/lib/utils/image";
import { sha256HexOfFile } from "@/lib/utils/crypto";
import { checkDuplicateQuick, checkDuplicateByImageHash } from "@/lib/utils/registry";
import { getFaceEmbedding, cosineSimilarity, countFaces } from "@/lib/utils/face";
import type { Hex } from "viem";

export function EnhancedAgentOrchestrator() {
  const chatAgent = useChatAgent();
  const swapAgent = useSwapAgent();
  const registerAgent = useRegisterIPAgent();
  const fileUpload = useFileUpload();
  const publicClient = usePublicClient();
  
  const [toast, setToast] = useState<string | null>(null);
  const [aiDetectionResult, setAiDetectionResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedFile, setAnalyzedFile] = useState<File | null>(null);
  const [lastDHash, setLastDHash] = useState<string | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [awaitingIdentity, setAwaitingIdentity] = useState<boolean>(false);
  const [dupCheck, setDupCheck] = useState<{ checked: boolean; found: boolean; tokenId?: string } | null>(null);
  const [refTemplates, setRefTemplates] = useState<Float32Array[] | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [showCamera, setShowCamera] = useState(false);

  const handleNewChat = useCallback(() => {
    chatAgent.newChat();
    try { fileUpload.removeFile(); } catch {}
    setAnalyzedFile(null);
    setAiDetectionResult(null);
    setReferenceFile(null);
    setAwaitingIdentity(false);
    setShowCamera(false);
    setLastDHash(null);
    setDupCheck(null);
    setToast(null);
  }, [chatAgent, fileUpload]);

  const handleOpenSession = useCallback((id: string) => {
    chatAgent.openSession(id);
    try { fileUpload.removeFile(); } catch {}
    setAnalyzedFile(null);
    setAiDetectionResult(null);
    setReferenceFile(null);
    setAwaitingIdentity(false);
    setShowCamera(false);
    setLastDHash(null);
    setDupCheck(null);
    setToast(null);
  }, [chatAgent, fileUpload]);

  const explorerBase = storyAeneid.blockExplorers?.default.url || "https://aeneid.storyscan.xyz";

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    chatScrollRef.current?.scrollTo({
      top: chatScrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [chatAgent.messages]);

  // Auto-analyze AI when file is uploaded
  useEffect(() => {
    if (!fileUpload.file) return;
    if (awaitingIdentity && referenceFile) {
      verifyIdentityPhoto(fileUpload.file).finally(() => {
        try { (fileInputRef.current as any)?.value && (fileInputRef.current!.value = ''); } catch {}
        try { (cameraInputRef.current as any)?.value && (cameraInputRef.current!.value = ''); } catch {}
        fileUpload.removeFile();
      });
      return;
    }
    if (!isAnalyzing) {
      analyzeImageForChat();
    }
  }, [fileUpload.file]);

  const analyzeImageForChat = async () => {
    if (!fileUpload.file) return;

    setIsAnalyzing(true);
    setAiDetectionResult(null);

    // Add immediate loading message when starting analysis
    const loadingMessage = {
      role: "agent" as const,
      text: "Wait a moment, let me analyze your image for AI and IP safety",
      ts: Date.now(),
      isLoading: true
    };

    chatAgent.addCompleteMessage(loadingMessage);

    // Store file reference before removing preview
    const currentFile = fileUpload.file;
    setAnalyzedFile(currentFile);

    // Create an object URL for preview image in chat (separate from upload preview)
    const previewUrl = URL.createObjectURL(currentFile);

    // Remove image preview from uploader immediately after upload
    setTimeout(() => {
      fileUpload.removeFile();
    }, 100);

    try {
      // Run AI detection, IP status, and whitelist in parallel
      const bufferPromise = fileToBuffer(currentFile);
      const aiPromise = bufferPromise.then((buffer) => detectAI(buffer));
      const mode = (process.env.NEXT_PUBLIC_IP_STATUS_MODE || 'client').toLowerCase();
      const clientIPText = 'Status: Local assessment\nRisk: Medium\nTolerance: Proceed with caution';
      const ipPromise = mode === 'server' ? detectIPStatus(currentFile) : Promise.resolve({ result: clientIPText });
      const wlPromise = isWhitelistedImage(currentFile);

      const [aiResultSettled, ipResultSettled, wlSettled] = await Promise.allSettled([aiPromise, ipPromise, wlPromise]);

      // Handle AI detection result
      let aiResult: { isAI: boolean; confidence: number } = { isAI: false, confidence: 0 };
      if (aiResultSettled.status === 'fulfilled') {
        aiResult = aiResultSettled.value as any;
        setAiDetectionResult({ ...aiResult, status: 'completed' });
      } else {
        setAiDetectionResult({ isAI: false, confidence: 0, status: 'failed' });
      }

      // Handle IP status result
      let ipText = "Status: Unknown\nRisk: Unable to determine\nTolerance: Good to register, please verify manually";
      if (ipResultSettled.status === 'fulfilled') {
        ipText = ipResultSettled.value.result;
      }

      // Whitelist override: If whitelisted, always treat as safe regardless of OpenAI assessment
      const wl = wlSettled.status === 'fulfilled' ? wlSettled.value : { whitelisted: false, reason: '', hash: '' };
      if (wl.whitelisted) {
        ipText = `Status: Looks suitable for registration.\nRisk: Low\nTolerance: Good to register`;
      }

      const aiText = aiResult.isAI
        ? `Analysis complete! Your image is AI generated with ${((aiResult.confidence || 0) * 100).toFixed(1)}% confidence.

Note: AI-generated images cannot be licensed for AI training purposes - it doesn't make sense to train AI with AI-generated content again!`
        : `Analysis complete! Your image appears to be real/human-made with ${((aiResult.confidence || 0) * 100).toFixed(1)}% confidence.`;

      // Normalize OpenAI response for non-whitelisted: don't allow "Good to register" unless Risk: Low
      if (!wl.whitelisted) {
        const lines = ipText.split('\n');
        const riskIdx = lines.findIndex(l => l.toLowerCase().startsWith('risk:'));
        const tolIdx = lines.findIndex(l => l.toLowerCase().startsWith('tolerance:'));
        const riskLineLower = (riskIdx >= 0 ? lines[riskIdx] : '').toLowerCase();
        const tolLower = (tolIdx >= 0 ? lines[tolIdx] : '').toLowerCase();
        const riskLow = riskLineLower.includes('low');
        const toleranceGood = tolLower.includes('good to register');
        if (toleranceGood && !riskLow && tolIdx >= 0) {
          lines[tolIdx] = 'Tolerance: Proceed with caution';
          ipText = lines.join('\n');
        }
        // Optional hard clamp for non-whitelisted (default true via env)
        const alwaysCaution = (process.env.NEXT_PUBLIC_NON_WL_ALWAYS_CAUTION ?? 'true') === 'true';
        if (alwaysCaution && tolIdx >= 0) {
          lines[tolIdx] = 'Tolerance: Proceed with caution';
          ipText = lines.join('\n');
        }
      }

      setLastDHash(wl.hash || null);
      const combinedText = `${aiText}\n\n${ipText}`;

      // Duplicate check (after safety analysis)
      let dupFound = false;
      let dupTokenId: string | undefined;
      try {
        const spg = process.env.NEXT_PUBLIC_SPG_COLLECTION as `0x${string}` | undefined;
        if (spg && publicClient) {
          const compressed = await compressImage(currentFile);
          const imageHash = (await sha256HexOfFile(compressed)).toLowerCase();
          const timeoutMs = Number.parseInt(process.env.NEXT_PUBLIC_REGISTRY_DUPCHECK_TIMEOUT_MS || '8000', 10);
          const withTimeout = <T,>(p: Promise<T>) => new Promise<T>((resolve) => {
            const t = setTimeout(() => resolve(null as any), timeoutMs);
            p.then(v => { clearTimeout(t); resolve(v); }).catch(() => { clearTimeout(t); resolve(null as any); });
          });
          const quick = await withTimeout(checkDuplicateQuick(publicClient, spg, imageHash));
          if (quick?.found) { dupFound = true; dupTokenId = quick.tokenId; }
          if (!dupFound) {
            const full = await withTimeout(checkDuplicateByImageHash(publicClient, spg, imageHash));
            if (full?.found) { dupFound = true; dupTokenId = full.tokenId; }
          }
        }
      } catch {}
      finally {
        setDupCheck({ checked: true, found: dupFound, tokenId: dupTokenId });
      }

      // Decide next action based on IP status tolerance/risk (whitelist forces safe)
      const riskLine = (ipText.split('\n').find(l => l.toLowerCase().startsWith('risk:')) || '').toLowerCase();
      const toleranceLineRaw = ipText.split('\n').find(l => l.toLowerCase().startsWith('tolerance:')) || '';
      const toleranceValue = toleranceLineRaw.split(':').slice(1).join(':').trim().toLowerCase();
      const riskLow = riskLine.includes('low');
      const toleranceGood = toleranceValue.startsWith('good to register');
      const isRisky = wl.whitelisted ? false : !(riskLow && toleranceGood);

      // Detect human face to offer camera capture option
      let faceDetected = false;
      try {
        // Prefer local FaceDetector API when available
        // @ts-ignore
        if (typeof window !== 'undefined' && window.FaceDetector) {
          // @ts-ignore
          const detector = new window.FaceDetector({ fastMode: true });
          const bitmap = await createImageBitmap(currentFile);
          const faces = await detector.detect(bitmap as any);
          faceDetected = Array.isArray(faces) && faces.length > 0;
        } else {
          // Fallback: keyword hints from OpenAI text
          const ipAll = ipText.toLowerCase();
          faceDetected = /face|faces|portrait|person|people|identity/.test(ipAll);
        }
      } catch {}

      // Identity requirement when analysis mentions identity/face
      const requiresIdentity = /identity|face|faces|portrait|person|people/.test(ipText.toLowerCase());
      if (requiresIdentity) {
        setReferenceFile(currentFile);
        setAwaitingIdentity(true);
      }

      // Compose buttons
      let buttons = dupFound ? ["Upload File", "Submit for Review", "Copy dHash"] : (isRisky ? ["Upload File", "Submit for Review", "Copy dHash"] : ["Continue Registration", "Copy dHash"]);
      if (faceDetected || requiresIdentity) {
        const cameraOnly = (process.env.NEXT_PUBLIC_CAMERA_ONLY_ON_FACE ?? 'false') === 'true';
        if (!buttons.includes("Take Photo")) buttons = ["Take Photo", ...buttons];
        if (cameraOnly) {
          buttons = buttons.filter(b => b !== "Upload File");
        }
      }
      if (requiresIdentity) {
        buttons = buttons.filter(b => b !== "Continue Registration");
      }

      // If duplicate, hide safe IP text and show remix tolerance guidance
      const duplicateBlockText = `\n\nDuplicate detected: this image is already registered as IP${dupTokenId ? ` (Token ID: ${dupTokenId})` : ''}. Registration is blocked.\nTolerance: Allowed to register as a remix`;
      const textToShow = dupFound ? `${aiText}${duplicateBlockText}` : combinedText;

      // Update the loading message to show results with appropriate next step and image preview
      chatAgent.updateLastMessage({
        text: textToShow,
        isLoading: false,
        buttons,
        image: { url: previewUrl, alt: currentFile.name }
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      setAiDetectionResult({
        isAI: false,
        confidence: 0,
        status: 'failed'
      });

      // Update loading message to show error
      const errorText = "❌ Sorry, I couldn't analyze the image. But don't worry, you can still proceed with registration!";

      chatAgent.updateLastMessage({
        text: errorText,
        isLoading: false
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const executePlan = useCallback(async () => {
    if (!chatAgent.currentPlan) return;

    const plan = chatAgent.currentPlan;

    if (plan.type === "swap" && plan.intent.kind === "swap") {
      chatAgent.updateStatus("🔄 Executing swap...");

      const result = await swapAgent.executeSwap(plan.intent);

      if (result.success) {
        const successMessage = `Swap success ✅
From: ${plan.intent.tokenIn}
To: ${plan.intent.tokenOut}
Amount: ${plan.intent.amount}
Tx: ${result.txHash}
↗ View: ${explorerBase}/tx/${result.txHash}`;

        chatAgent.addMessage("agent", successMessage);
        setToast("Swap success ✅");
      } else {
        chatAgent.addMessage("agent", `Swap error: ${result.error}`);
        setToast("Swap error ❌");
      }
      
      chatAgent.clearPlan();
      swapAgent.resetSwap();
    }
    
    else if (plan.type === "register" && plan.intent.kind === "register") {
      // Get file from engine context or fallback to analyzed file
      const fileToUse = chatAgent.getEngineFile() || analyzedFile;

      if (!fileToUse) {
        chatAgent.addMessage("agent", "❌ Please attach an image first!");
        setToast("Attach image first 📎");
        return;
      }

      // Duplicate check before signing (skip if already checked safe during analysis)
      const alreadyCheckedSafe = dupCheck?.checked && !dupCheck.found;
      if (!alreadyCheckedSafe) {
        try {
          const { compressImage } = await import("@/lib/utils/image");
          const { sha256HexOfFile } = await import("@/lib/utils/crypto");
          const { checkDuplicateByImageHash, checkDuplicateQuick } = await import("@/lib/utils/registry");
          const spg = process.env.NEXT_PUBLIC_SPG_COLLECTION as `0x${string}` | undefined;
          if (spg && publicClient) {
            const compressed = await compressImage(fileToUse);
            const imageHash = (await sha256HexOfFile(compressed)).toLowerCase();
            const timeoutMs = Number.parseInt(process.env.NEXT_PUBLIC_REGISTRY_DUPCHECK_TIMEOUT_MS || '8000', 10);
            const withTimeout = <T,>(p: Promise<T>) => new Promise<T>((resolve, reject) => {
              const t = setTimeout(() => resolve(null as any), timeoutMs);
              p.then(v => { clearTimeout(t); resolve(v); }).catch(e => { clearTimeout(t); reject(e); });
            });
            const quick = await withTimeout(checkDuplicateQuick(publicClient, spg, imageHash));
            if (quick?.found) {
              chatAgent.addMessage("agent", `❌ This image is already registered as IP (Token ID: ${quick.tokenId}). Registration blocked.`);
              setToast("Duplicate image detected ❌");
              chatAgent.clearPlan();
              return;
            }
            const dup = await withTimeout(checkDuplicateByImageHash(publicClient, spg, imageHash));
            if (dup?.found) {
              chatAgent.addMessage("agent", `❌ This image is already registered as IP (Token ID: ${dup.tokenId}). Registration blocked.`);
              setToast("Duplicate image detected ❌");
              chatAgent.clearPlan();
              return;
            }
          }
        } catch (e) {
          console.warn("Duplicate pre-check failed:", e);
        }
      }

      chatAgent.updateStatus("📝 Registering IP...");

      // Use default license settings from the plan
      const licenseSettings: LicenseSettings = {
        ...DEFAULT_LICENSE_SETTINGS,
        pilType: plan.intent.pilType || DEFAULT_LICENSE_SETTINGS.pilType,
      };

      const result = await registerAgent.executeRegister(plan.intent, fileToUse, licenseSettings);

      if (result.success) {
        // Show initial success with transaction link
        const submittedMessage = `Tx submitted ���\n↗ View: ${explorerBase}/tx/${result.txHash}`;
        chatAgent.addMessage("agent", submittedMessage);

        // Wait for confirmation
        try {
          chatAgent.updateStatus("Waiting for confirmation...");
          const confirmed = await waitForTxConfirmation(
            publicClient, 
            result.txHash as Hex,
            { timeoutMs: 90_000 }
          );

          if (confirmed) {
            const successText = `Register success ✅

Your image has been successfully registered as IP!

License Type: ${result.licenseType}
AI Detected: ${result.aiDetected ? 'Yes' : 'No'} (${((result.aiConfidence || 0) * 100).toFixed(1)}%)`;

            // Create message with image and links
            const message = {
              role: "agent" as const,
              text: successText,
              ts: Date.now(),
              image: result.imageUrl ? {
                url: result.imageUrl,
                alt: "Registered IP image"
              } : undefined,
              links: [
                {
                  text: `📋 View IP: ${result.ipId}`,
                  url: `https://aeneid.explorer.story.foundation/ipa/${result.ipId}`
                },
                {
                  text: `🔗 View Transaction: ${result.txHash}`,
                  url: `${explorerBase}/tx/${result.txHash}`
                }
              ]
            };

            chatAgent.addCompleteMessage(message);
            setToast("IP registered ✅");
          } else {
            chatAgent.updateStatus("Tx still pending on network. Check explorer.");
          }
        } catch {
          chatAgent.updateStatus("Tx still pending on network. Check explorer.");
        }
      } else {
        chatAgent.addMessage("agent", `Register error: ${result.error}`);
        setToast("Register error ❌");
      }
      
      chatAgent.clearPlan();
      registerAgent.resetRegister();
      setAnalyzedFile(null);
      setAiDetectionResult(null);
    }
  }, [
    chatAgent,
    swapAgent,
    registerAgent,
    analyzedFile,
    publicClient,
    explorerBase,
    aiDetectionResult
  ]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Prepare multi-shot reference templates when identity is required
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (awaitingIdentity && referenceFile) {
        try {
          chatAgent.updateStatus('Preparing identity templates...');
          const { getAugmentedEmbeddings } = await import('@/lib/utils/face');
          const tmpls = await getAugmentedEmbeddings(referenceFile, [-25, -15, 0, 15, 25], [false, true]);
          if (!cancelled) setRefTemplates(tmpls);
        } catch {
          if (!cancelled) setRefTemplates(null);
        }
      } else {
        setRefTemplates(null);
      }
    })();
    return () => { cancelled = true; };
  }, [awaitingIdentity, referenceFile, chatAgent]);

  const handleButtonClick = useCallback((buttonText: string) => {
    if (buttonText === "Register IP") {
      // Start register flow by asking for file directly (no chat prompt)
      fileInputRef.current?.click();
      return;
    }
    if (buttonText === "Upload File") {
      fileInputRef.current?.click();
    } else if (buttonText === "Continue Registration") {
      chatAgent.processPrompt(buttonText, (referenceFile || analyzedFile) || undefined, aiDetectionResult);
    } else if (buttonText === "Take Photo") {
      if (!referenceFile && analyzedFile) setReferenceFile(analyzedFile);
      setAwaitingIdentity(true);
      setShowCamera(true);
    } else if (buttonText === "Submit for Review") {
      const email = "apilpirman@gmail.com";
      const subject = encodeURIComponent("IP Review Request");
      const body = encodeURIComponent(
        `Hello,

I would like to submit my IP asset for manual review with permissions.

Included (recommended):
- Proof of ownership or authorization letter
- License/permission documents
- Source references and links
- Contact info (name, wallet address)

Thank you.`
      );
      const mailto = `mailto:${email}?subject=${subject}&body=${body}`;
      chatAgent.addCompleteMessage({
        role: "agent",
        text: `This asset may be risky. Please submit your documents for manual review via email: ${email}\nAttach authorization letters, license proofs, ownership evidence, references, and your contact info.`,
        ts: Date.now(),
        links: [{ text: "Open email to submit documents", url: mailto }]
      });
    } else if (buttonText === "Copy dHash") {
      if (lastDHash) {
        navigator.clipboard.writeText(lastDHash).then(() => {
          setToast("dHash copied ✅");
        }).catch(() => setToast("Copy failed ❌"));
      } else {
        setToast("No dHash available ❌");
      }
    } else {
      chatAgent.processPrompt(buttonText);
    }
  }, [chatAgent, analyzedFile, aiDetectionResult, lastDHash]);

  const handleFileInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      fileUpload.handleFileSelect(file);
      // Reset input for re-selection of same file
      event.target.value = '';
    }
  }, [fileUpload]);

  const verifyIdentityPhoto = useCallback(async (capture: File) => {
    if (!referenceFile) return;
    try {
      chatAgent.updateStatus('Verifying identity with camera photo...');

      // Multi-face check on capture
      try {
        const faces = await countFaces(capture);
        if (faces > 1) {
          setToast('Multiple faces detected ❌');
          chatAgent.addMessage('agent', 'Multiple faces detected in the photo. Please retake with only one face clearly visible.', ['Take Photo', 'Submit for Review']);
          return;
        }
      } catch {}

      // Advanced embedding comparison with multi-augmentation (robust to pose/rotation/flip)
      const simTh = parseFloat(process.env.NEXT_PUBLIC_FACE_SIM_THRESHOLD || '0.86');
      try {
        const { getAugmentedEmbeddings, cosineSimilarity, compareFacesAdvanced } = await import("@/lib/utils/face");
        if (refTemplates && refTemplates.length > 0) {
          const probe = await getAugmentedEmbeddings(capture, [-25, -15, 0, 15, 25], [false, true]);
          let best = 0;
          for (const a of refTemplates) for (const b of probe) best = Math.max(best, cosineSimilarity(a, b));
          if (best >= simTh) {
            setAwaitingIdentity(false);
            setToast('Identity verified ✅');
            chatAgent.addMessage('agent', `Identity verified (best similarity ${best.toFixed(3)} ≥ ${simTh}). Proceeding to registration.`);
            chatAgent.processPrompt('Continue Registration', referenceFile, aiDetectionResult);
            return;
          } else {
            setToast('Identity mismatch ❌');
            chatAgent.addMessage('agent', `Identity check failed (best similarity ${best.toFixed(3)} < ${simTh}). Try again with clearer, front-facing photo.`);
            chatAgent.addMessage('agent', 'You can take another photo or submit for review.', ['Take Photo', 'Submit for Review']);
            return;
          }
        }
        // Fallback if templates not ready
        const { best } = await compareFacesAdvanced(referenceFile, capture, { rotations: [-25, -15, 0, 15, 25], allowFlip: true });
        if (best >= simTh) {
          setAwaitingIdentity(false);
          setToast('Identity verified ✅');
          chatAgent.addMessage('agent', `Identity verified (best similarity ${best.toFixed(3)} ≥ ${simTh}). Proceeding to registration.`);
          chatAgent.processPrompt('Continue Registration', referenceFile, aiDetectionResult);
          return;
        } else {
          setToast('Identity mismatch ❌');
          chatAgent.addMessage('agent', `Identity check failed (best similarity ${best.toFixed(3)} < ${simTh}). Try again with clearer, front-facing photo.`);
          chatAgent.addMessage('agent', 'You can take another photo or submit for review.', ['Take Photo', 'Submit for Review']);
          return;
        }
      } catch {}

      // Fallback: perceptual dHash if face embedding not available
      const hashSize = Number.parseInt(process.env.NEXT_PUBLIC_SAFE_IMAGE_DHASH_SIZE || '8', 10);
      const cropEnv = parseFloat(process.env.NEXT_PUBLIC_SAFE_IMAGE_CENTER_CROP || '0.7');
      const crop = Math.max(0.4, Math.min(0.95, isNaN(cropEnv) ? 0.7 : cropEnv));

      const refBase = await computeDHash(referenceFile, hashSize, false, 1);
      const refFlip = await computeDHash(referenceFile, hashSize, true, 1);
      const refCenter = await computeDHash(referenceFile, hashSize, false, crop);
      const refCenterFlip = await computeDHash(referenceFile, hashSize, true, crop);

      const capBase = await computeDHash(capture, hashSize, false, 1);
      const capFlip = await computeDHash(capture, hashSize, true, 1);
      const capCenter = await computeDHash(capture, hashSize, false, crop);
      const capCenterFlip = await computeDHash(capture, hashSize, true, crop);

      const refs = [refBase, refFlip, refCenter, refCenterFlip];
      const caps = [capBase, capFlip, capCenter, capCenterFlip];
      const hDist = (a: string, b: string) => {
        const len = Math.min(a.length, b.length);
        let dist = 0;
        for (let i = 0; i < len; i++) {
          const x = (parseInt(a[i], 16) ^ parseInt(b[i], 16)) & 0xf;
          dist += [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4][x];
        }
        dist += Math.abs(a.length - b.length) * 4;
        return dist;
      };
      let best = Infinity;
      for (const r of refs) for (const c of caps) best = Math.min(best, hDist(r, c));

      const th = Number.parseInt(process.env.NEXT_PUBLIC_IDENTITY_DHASH_THRESHOLD || '14', 10);
      if (best <= th) {
        setAwaitingIdentity(false);
        setToast('Identity verified ✅');
        chatAgent.addMessage('agent', `Identity verified (distance ${best} ≤ ${th}). Proceeding to registration.`);
        chatAgent.processPrompt('Continue Registration', referenceFile, aiDetectionResult);
      } else {
        setToast('Identity mismatch ❌');
        chatAgent.addMessage('agent', `Identity check failed (distance ${best} > ${th}). Please retake photo or upload proof.`);
        chatAgent.addMessage('agent', 'You can take another photo or submit for review.', ['Take Photo', 'Submit for Review']);
      }
    } catch (e) {
      setToast('Identity check error ❌');
      chatAgent.addMessage('agent', 'Identity verification encountered an error. Please try again.');
    }
  }, [referenceFile, aiDetectionResult, chatAgent]);

  return (
    <div className="mx-auto max-w-[1400px] px-2 sm:px-4 md:px-6 overflow-x-hidden">
      <div className="flex flex-col lg:grid lg:grid-cols-[180px,1fr] gap-3 lg:gap-6 h-[calc(100vh-120px)] lg:h-[calc(100vh-180px)]">
        {/* History Sidebar - Hidden on mobile, shown on desktop */}
        <div className="hidden lg:block">
          <HistorySidebar
            messages={chatAgent.messages}
            onNewChat={handleNewChat}
            chatHistory={chatAgent.history?.map(h => ({ id: h.id, title: h.title, lastMessage: h.lastMessage, timestamp: h.timestamp, messageCount: h.messageCount }))}
            onOpenSession={handleOpenSession}
          />
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          {/* Header */}
          <div className="shrink-0 mb-3 lg:mb-4">
            <div className="flex items-center justify-between rounded-xl bg-white/5 border border-white/10 p-3 lg:p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-sky-400 to-blue-500 flex items-center justify-center shadow-lg">
                  <span className="text-lg font-bold text-white">S</span>
                </div>
                <div>
                  <div className="text-sm font-semibold text-white">CHAT WITH SUPERLEE</div>
                </div>
              </div>

              {/* Mobile menu button for history */}
              <button
                onClick={handleNewChat}
                className="lg:hidden p-2 rounded-lg bg-white/10 hover:bg-white/15 transition-colors"
                title="New Chat"
              >
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>
          </div>

          {/* Chat Content */}
          <section className="flex-1 rounded-2xl border border-white/10 bg-white/5 overflow-hidden flex flex-col min-h-0">
            {/* Messages Area */}
            <div
              ref={chatScrollRef}
              className="flex-1 overflow-y-auto scrollbar-invisible"
            >
              <div className="mx-auto w-full max-w-[900px] px-2 sm:px-3 lg:px-4 py-3 lg:py-4">
                <MessageList
                  messages={chatAgent.messages}
                  onButtonClick={handleButtonClick}
                  isTyping={chatAgent.isTyping}
                />


                {/* Plan Box */}
                {chatAgent.currentPlan && (
                  <PlanBox
                    plan={chatAgent.currentPlan}
                    onConfirm={executePlan}
                    onCancel={chatAgent.clearPlan}
                    swapState={swapAgent.swapState}
                    registerState={registerAgent.registerState}
                  />
                )}
              </div>
            </div>

            {/* Composer */}
            <div className="shrink-0">
              <Composer
                onSubmit={(prompt) => chatAgent.processPrompt(prompt, fileUpload.file || undefined, aiDetectionResult)}
                status={chatAgent.status}
                file={fileUpload.file}
                onFileSelect={fileUpload.handleFileSelect}
                onFileRemove={fileUpload.removeFile}
                previewUrl={fileUpload.previewUrl}
                isTyping={chatAgent.isTyping}
                awaitingInput={chatAgent.awaitingInput}
                messages={chatAgent.messages}
              />
            </div>
          </section>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="user"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      {/* Camera Modal */}
      <CameraCapture
        open={showCamera}
        onClose={() => setShowCamera(false)}
        onCapture={(file) => {
          fileUpload.handleFileSelect(file);
          // verification will kick via useEffect when awaitingIdentity=true
        }}
        onFallback={() => {
          setShowCamera(false);
          cameraInputRef.current?.click();
        }}
      />

      {/* Toast Notifications */}
      <Toast
        message={toast}
        onClose={() => setToast(null)}
      />
    </div>
  );
}
