import pdb
import logging

from dotenv import load_dotenv
load_dotenv()

import os
import glob
import asyncio
import argparse
import json  # for API output serialization

logger = logging.getLogger(__name__)

import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

# --------------------- FASTAPI IMPORTS & SETUP ---------------------
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Browser Agent API")
lock = asyncio.Lock()

class RunRequest(BaseModel):
    agent_type: str
    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_base_url: str
    llm_api_key: str
    use_own_browser: bool
    keep_browser_open: bool
    headless: bool
    disable_security: bool
    window_w: int
    window_h: int
    save_recording_path: str
    save_agent_history_path: str
    save_trace_path: str
    enable_recording: bool
    task: str
    add_infos: str
    max_steps: int
    use_vision: bool
    max_actions_per_step: int
    tool_calling_method: str

class WrappedParams(BaseModel):
    new_store_name: str
    new_owner_email: str

@app.post("/wrapped_run_custom_agent")
async def wrapped_run_custom_agent_api(params: WrappedParams):
    open_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    
    shopify_partner_id = os.environ.get("SHOPIFY_PARTNER_ID")
    if not shopify_partner_id:
        raise ValueError("SHOPIFY_PARTNER_ID environment variable not found.")
    
    shopify_partner_email = os.environ.get("SHOPIFY_PARTNER_EMAIL")
    if not shopify_partner_email:
        raise ValueError("SHOPIFY_PARTNER_EMAIL environment variable not found.")
    
    shopify_partner_password = os.environ.get("SHOPIFY_PARTNER_PASSWORD", "Dollar@2025")

    # Fixed JSON payload as provided.
    payload = {
        "agent_type": "custom",
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "llm_temperature": 1,
        "llm_base_url": "",
        "llm_api_key": open_api_key,
        "use_own_browser": True,
        "keep_browser_open": True,
        "headless": False,
        "disable_security": True,
        "window_w": 1280,
        "window_h": 1100,
        "save_recording_path": "./tmp/record_videos",
        "save_agent_history_path": "./tmp/agent_history",
        "save_trace_path": "./tmp/traces",
        "enable_recording": True,
        "task": (
            "Phase 1: \nOpen a new tab to https://partners.shopify.com/4110997/stores/new?store_type=client_store\n"
            "If asked, Login with kaduriraghd@hotmail.com\nDollar@2025. \nRe-enter store name NhatTestStore000030\n"
            "If store name is unavailable, complete \nClick on online only, and this customer is not from another tool.\n\n"
            "Else, click on 'create development store'. If a redirect happens, you are good to open a new tab to "
            "https://admin.shopify.com/store/{store_name}/settings/account?transfer_ownership=true\nIf it takes you to "
            "another link, just click on the first account. Then, when redirected, click on 'continue with unsupported browser'.\n"
            "For new store owner, enter:\nEmail: nhatvhn99@gmail.com\nFirst Name: Nhat\nLast Name: Vu\n"
            "Password: Dollar@2025\nClick 'Transfer Store Ownership'. Done. Don't do anything else."
        ),
        "add_infos": "",
        "max_steps": 100,
        "use_vision": False,
        "max_actions_per_step": 10,
        "tool_calling_method": "auto"
    }

    # Replace the hardcoded store name and owner email with the provided values.
    modified_task = payload["task"].replace("NhatTestStore000030", params.new_store_name) \
                                   .replace("nhatvhn99@gmail.com", params.new_owner_email) \
                                   .replace("4110997", shopify_partner_id) \
                                   .replace("kaduriraghd@hotmail.com", shopify_partner_email) \
                                   .replace("Dollar@2025", shopify_partner_password)
    payload["task"] = modified_task

    # Construct the LLM model.
    llm = utils.get_llm_model(
        provider=payload["llm_provider"],
        model_name=payload["llm_model_name"],
        temperature=payload["llm_temperature"],
        base_url=payload["llm_base_url"],
        api_key=payload["llm_api_key"],
    )

    # Call run_custom_agent with parameters from the payload.
    result = await run_custom_agent(
        llm=llm,
        use_own_browser=payload["use_own_browser"],
        keep_browser_open=payload["keep_browser_open"],
        headless=payload["headless"],
        disable_security=payload["disable_security"],
        window_w=payload["window_w"],
        window_h=payload["window_h"],
        save_recording_path=payload["save_recording_path"],
        save_agent_history_path=payload["save_agent_history_path"],
        save_trace_path=payload["save_trace_path"],
        task=payload["task"],
        add_infos=payload["add_infos"],
        max_steps=payload["max_steps"],
        use_vision=payload["use_vision"],
        max_actions_per_step=payload["max_actions_per_step"],
        tool_calling_method=payload["tool_calling_method"]
    )

    # run_custom_agent returns:
    # (final_result, errors, model_actions, model_thoughts, trace_file, history_file)
    return {
        "final_result": result[0],
        "errors": result[1],
        "model_actions": result[2],
        "model_thoughts": result[3],
        "trace_file": result[4],
        "history_file": result[5]
    }

@app.post("/run_with_stream")
async def run_with_stream_api(req: RunRequest):
    """
    POST endpoint to call the run_with_stream function.
    It streams the outputs (JSON-serialized) as they are yielded.
    """
    async def event_generator():
        # Call your existing run_with_stream function with parameters from the request.
        async for output in run_with_stream(
            agent_type=req.agent_type,
            llm_provider=req.llm_provider,
            llm_model_name=req.llm_model_name,
            llm_temperature=req.llm_temperature,
            llm_base_url=req.llm_base_url,
            llm_api_key=req.llm_api_key,
            use_own_browser=req.use_own_browser,
            keep_browser_open=req.keep_browser_open,
            headless=req.headless,
            disable_security=req.disable_security,
            window_w=req.window_w,
            window_h=req.window_h,
            save_recording_path=req.save_recording_path,
            save_agent_history_path=req.save_agent_history_path,
            save_trace_path=req.save_trace_path,
            enable_recording=req.enable_recording,
            task=req.task,
            add_infos=req.add_infos,
            max_steps=req.max_steps,
            use_vision=req.use_vision,
            max_actions_per_step=req.max_actions_per_step,
            tool_calling_method=req.tool_calling_method
        ):
            # Use json.dumps with default=str to ensure non-serializable objects (e.g. gr.update) are handled.
            yield json.dumps(output, default=str) + "\n"
    return StreamingResponse(event_generator(), media_type="application/json")
@app.post("/run_custom_agent")
async def run_custom_agent_api(req: RunRequest):
    """
    POST endpoint to call the run_custom_agent function directly.
    It returns a JSON response containing:
      - final_result
      - errors
      - model_actions
      - model_thoughts
      - trace_file
      - history_file
    """
    # Create the LLM model using the parameters provided in the request
    llm = utils.get_llm_model(
        provider=req.llm_provider,
        model_name=req.llm_model_name,
        temperature=req.llm_temperature,
        base_url=req.llm_base_url,
        api_key=req.llm_api_key,
    )
    
    # Call run_custom_agent with the provided parameters
    result = await run_custom_agent(
        llm=llm,
        use_own_browser=req.use_own_browser,
        keep_browser_open=req.keep_browser_open,
        headless=req.headless,
        disable_security=req.disable_security,
        window_w=req.window_w,
        window_h=req.window_h,
        save_recording_path=req.save_recording_path,
        save_agent_history_path=req.save_agent_history_path,
        save_trace_path=req.save_trace_path,
        task=req.task,
        add_infos=req.add_infos,
        max_steps=req.max_steps,
        use_vision=req.use_vision,
        max_actions_per_step=req.max_actions_per_step,
        tool_calling_method=req.tool_calling_method
    )
    
    # run_custom_agent returns a tuple:
    # (final_result, errors, model_actions, model_thoughts, trace_file, history_file)
    return {
        "final_result": result[0],
        "errors": result[1],
        "model_actions": result[2],
        "model_thoughts": result[3],
        "trace_file": result[4],
        "history_file": result[5]
    }
# --------------------- END FASTAPI SETUP ---------------------


# Global variables for persistence
_global_browser = None
_global_browser_context = None

# Create the global agent state instance
_global_agent_state = AgentState()

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',                                         # final_result
            errors,                                     # errors
            '',                                         # model_actions
            '',                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # history_file
            None,                                       # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    async with lock:    
        try:
            global _global_browser, _global_browser_context, _global_agent_state
            
            # Clear any previous stop request
            _global_agent_state.clear_stop()

            extra_chromium_args = [f"--window-size={window_w},{window_h}"]
            if use_own_browser:
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
            else:
                chrome_path = None
                
            if _global_browser is None:
                _global_browser = Browser(
                    config=BrowserConfig(
                        headless=headless,
                        disable_security=disable_security,
                        chrome_instance_path=chrome_path,
                        extra_chromium_args=extra_chromium_args,
                    )
                )

            if _global_browser_context is None:
                _global_browser_context = await _global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path=save_trace_path if save_trace_path else None,
                        save_recording_path=save_recording_path if save_recording_path else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(
                            width=window_w, height=window_h
                        ),
                    )
                )
                
            agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
            history = await agent.run(max_steps=max_steps)

            history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
            agent.save_history(history_file)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()

            trace_file = get_latest_files(save_trace_path)

            return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors = str(e) + "\n" + traceback.format_exc()
            return '', errors, '', '', None, None
        finally:
            # Handle cleanup based on persistence configuration
            if not keep_browser_open:
                if _global_browser_context:
                    await _global_browser_context.close()
                    _global_browser_context = None

                if _global_browser:
                    await _global_browser.close()
                    _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    async with lock:
        try:
            global _global_browser, _global_browser_context, _global_agent_state

            # Clear any previous stop request
            _global_agent_state.clear_stop()

            extra_chromium_args = [f"--window-size={window_w},{window_h}"]
            if use_own_browser:
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
            else:
                chrome_path = None

            controller = CustomController()

            # Initialize global browser if needed
            if _global_browser is None:
                _global_browser = CustomBrowser(
                    config=BrowserConfig(
                        headless=headless,
                        disable_security=disable_security,
                        chrome_instance_path=chrome_path,
                        extra_chromium_args=extra_chromium_args,
                    )
                )

            if _global_browser_context is None:
                _global_browser_context = await _global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path=save_trace_path if save_trace_path else None,
                        save_recording_path=save_recording_path if save_recording_path else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(
                            width=window_w, height=window_h
                        ),
                    )
                )
                
            # Create and run agent
            agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                agent_state=_global_agent_state,
                tool_calling_method=tool_calling_method
            )
            history = await agent.run(max_steps=max_steps)

            history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
            agent.save_history(history_file)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()

            trace_file = get_latest_files(save_trace_path)        

            return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors = str(e) + "\n" + traceback.format_exc()
            return '', errors, '', '', None, None
        finally:
            # Handle cleanup based on persistence configuration
            if not keep_browser_open:
                if _global_browser_context:
                    await _global_browser_context.close()
                    _global_browser_context = None

                if _global_browser:
                    await _global_browser.close()
                    _global_browser = None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None

            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stop", interactive=True),  # Re-enable stop button
                        gr.update(interactive=True)  # Re-enable run button
                    ]
                await asyncio.sleep(0.05)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                latest_videos = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)    # Re-enable run button
            ]

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}

async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None

def create_ui(config, theme_name="Ocean"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    js = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css, js=js
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value=config['agent_type'],
                        info="Select the type of agent to use",
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=config['use_vision'],
                            info="Enable visual processing capabilities",
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value=config['tool_calling_method'],
                            interactive=True,
                            allow_custom_value=True,  # Allow users to input custom model names
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Funtion Name",
                            visible=False
                        )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider,model in utils.model_names.items()],
                        label="LLM Provider",
                        value=config['llm_provider'],
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names['openai'],
                        value=config['llm_model_name'],
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select a model from the dropdown or type a custom model name"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=config['llm_base_url'],
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=config['llm_api_key'],
                            info="Your API key (leave blank to use .env)"
                        )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=config['use_own_browser'],
                            info="Use your existing browser instance",
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=config['keep_browser_open'],
                            info="Keep Browser Open between Tasks",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=config['headless'],
                            info="Run browser without GUI",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=config['disable_security'],
                            info="Disable browser security features",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=config['enable_recording'],
                            info="Enable saving browser recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=config['window_w'],
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=config['window_h'],
                            info="Browser window height",
                        )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value=config['task'],
                    info="Describe what you want the agent to do",
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    
                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                )

            with gr.TabItem("📁 Configuration", id=5):
                with gr.Group():
                    config_file_input = gr.File(
                        label="Load Config File",
                        file_types=[".pkl"],
                        interactive=True
                    )

                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")

                    config_status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )

                load_config_button.click(
                    fn=update_ui_from_config,
                    inputs=[config_file_input],
                    outputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                        window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                        task, config_status
                    ]
                )

                save_config_button.click(
                    fn=save_current_config,
                    inputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                        save_agent_history_path, task,
                    ],  
                    outputs=[config_status]
                )

            with gr.TabItem("📊 Results", id=6):
                with gr.Group():

                    recording_display = gr.Video(label="Latest Recording")

                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result", lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors", lines=3, show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions", lines=3, show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts", lines=3, show_label=True
                            )

                    trace_file = gr.File(label="Trace File")

                    agent_history_file = gr.File(label="Agent History")

                # Bind the stop button click event after errors_output is defined
                stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[errors_output, stop_button, run_button],
                )

                # Run button click handler
                run_button.click(
                    fn=run_with_stream,
                        inputs=[
                            agent_type, llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                            save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                            enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_calling_method
                        ],
                    outputs=[
                        browser_view,           # Browser view
                        final_result_output,    # Final result
                        errors_output,          # Errors
                        model_actions_output,   # Model actions
                        model_thoughts_output,  # Model thoughts
                        recording_display,      # Latest recording
                        trace_file,             # Trace file
                        agent_history_file,     # Agent history file
                        stop_button,            # Stop button
                        run_button              # Run button
                    ],
                )

            with gr.TabItem("🎥 Recordings", id=7):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings(config['save_recording_path']),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI / FastAPI for Browser Agent")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on for the Gradio UI")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    # New arguments for API mode:
    parser.add_argument("--api", action="store_true", help="Run as API server (FastAPI) instead of launching the UI")
    parser.add_argument("--api-port", type=int, default=8000, help="Port to run the API server on")
    args = parser.parse_args()

    config_dict = default_config()

    if args.api:
        # Run the FastAPI server.
        # (You can start this with: python your_script.py --api)
        import uvicorn
        uvicorn.run(app, host=args.ip, port=args.api_port)
    else:
        demo = create_ui(config_dict, theme_name=args.theme)
        demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
