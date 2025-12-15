# cvat

## hack to hide false attributes

```js
if (window.falseObserver) {
    window.falseObserver.disconnect();
}

function highlightFalseAttributes() {
    document.querySelectorAll('#cvat_canvas_text_content tspan.cvat_canvas_text_attribute').forEach(tspan => {
        const childTspan = tspan.querySelector('tspan');
        if (childTspan) {
            const val = childTspan.textContent.trim();
            tspan.style.opacity = (val === 'false') ? '0' : '1';
        } else {
            const text = tspan.textContent.trim();
            if (text.endsWith('false')) {
                tspan.style.opacity = '0';
            } else {
                tspan.style.opacity = '1';
            }
        }
    });
}

highlightFalseAttributes();

window.falseObserver = new MutationObserver(() => {
    highlightFalseAttributes();
});

const targetNode = document.querySelector('#cvat_canvas_text_content');
if (targetNode) {
    window.falseObserver.observe(targetNode, { childList: true, subtree: true });
}
```
