# chatbot-integration-demo
Simple website to use as a harness for the AI chatbot (separate project).



# Local Development Setup

## Prerequisites

- **Node.js** 16+ and npm 8+
- **Docker** and Docker Compose (for full stack)
- **Git** for version control

## Quick Start (Test Harness Only)

Get the chatbot widget test harness running in under 5 minutes:

```bash
# Clone the repository
git clone https://github.com/yourorg/chatbot-integration-demo.git
cd chatbot-integration-demo

# Install dependencies
npm install

# Start the test server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the widget integration demos.

## Full Development Setup

### 1. Environment Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Basic settings
NODE_ENV=development
PORT=3000

# Widget defaults
DEFAULT_WIDGET_THEME=corporate-blue
DEFAULT_WIDGET_POSITION=bottom-right
DEFAULT_GREETING="Hello! How can I help you today?"

# Demo site configurations
CORPORATE_DEMO_NAME="Acme Corporation"
ECOMMERCE_DEMO_NAME="TechShop Online"
```

### 2. Start Development Services

#### Option A: Test Harness Only (Lightweight)
```bash
npm run dev
```

#### Option B: Full Stack with Docker (Complete Platform)
```bash
# Start all services (API, Redis, PostgreSQL, Vector DB)
docker-compose up -d

# Install widget dependencies
npm install

# Start the development server
npm run dev:full
```

### 3. Available Development URLs

| Service | URL | Description |
|---------|-----|-------------|
| Test Harness | [http://localhost:3000](http://localhost:3000) | Widget integration demos |
| CDN Integration | [http://localhost:3000/](http://localhost:3000/) | Script tag integration test |
| NPM Integration | [http://localhost:3000/npm-test](http://localhost:3000/npm-test) | Package integration test |
| Iframe Integration | [http://localhost:3000/iframe-test](http://localhost:3000/iframe-test) | Iframe embedding test |
| Mobile Test | [http://localhost:3000/mobile-test](http://localhost:3000/mobile-test) | Responsive design test |
| Corporate Demo | [http://localhost:3000/corporate-sim](http://localhost:3000/corporate-sim) | Corporate website simulation |
| E-commerce Demo | [http://localhost:3000/ecommerce-sim](http://localhost:3000/ecommerce-sim) | E-commerce integration |

## Development Commands

```bash
# Development
npm run dev              # Start test harness server
npm run dev:watch        # Start with auto-reload
npm run dev:full         # Start with full backend stack

# Testing
npm test                 # Run unit tests
npm run test:watch       # Run tests in watch mode
npm run test:e2e         # Run end-to-end tests

# Code Quality
npm run lint             # Check code style
npm run lint:fix         # Fix linting issues
npm run format           # Format code with Prettier

# Build
npm run build            # Build for production
npm run build:examples   # Build integration examples
```

## Project Structure

```
chatbot-integration-demo/
├── test-server/         # Express.js test harness
│   ├── app.js          # Main server application
│   ├── views/          # EJS templates for test pages
│   └── public/         # Static assets (CSS, JS, images)
├── examples/           # Real-world integration examples
├── mock-data/          # Sample data for testing
├── documentation/      # Integration guides
└── scripts/           # Utility scripts
```

## Testing Your Integration

### 1. Test Widget Functionality
Visit the test pages to verify widget behavior:
- **Basic functionality**: Message sending, receiving, UI interactions
- **Theme switching**: Corporate Blue, Warm Orange, Modern Purple
- **Responsive design**: Mobile, tablet, desktop layouts
- **Integration methods**: CDN, NPM, iframe embedding

### 2. Test API Endpoints
```bash
# Test chat endpoint
curl -X POST http://localhost:3000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "sessionId": "test-session"}'

# Test health check
curl http://localhost:3000/health
```

### 3. Integration Examples
Check the `examples/` directory for real-world integration code:
- `vanilla-html/` - Pure HTML/JS integration
- `react-app/` - React component integration  
- `wordpress-plugin/` - WordPress plugin example

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm run dev
```

**Missing dependencies:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Docker issues:**
```bash
# Reset Docker environment
docker-compose down -v
docker-compose up -d --build
```

### Getting Help

- **Widget not loading**: Check browser console for errors
- **API errors**: Check server logs with `npm run dev` 
- **Style conflicts**: Test in different themes and browsers
- **Integration issues**: Reference examples in `examples/` directory

## Next Steps

1. **Explore Integration Methods**: Try CDN, NPM, and iframe approaches
2. **Customize Themes**: Modify CSS themes in `test-server/public/css/themes/`
3. **Test Responsive Design**: Use mobile test page and browser dev tools
4. **Build Your Integration**: Use examples as starting templates

For more detailed integration instructions, see the [Integration Guide](documentation/integration-guide.md).

# Widget Architecture: Web Components + Iframe Fallback

## Overview

Our AI chatbot widget uses a **hybrid rendering architecture** that combines modern Web Components with iframe fallback to deliver optimal performance across all browser environments while maintaining complete style isolation from host websites.

## The Challenge

When embedding third-party widgets on customer websites, developers face a fundamental trade-off:

- **Direct DOM manipulation** offers great performance but suffers from CSS conflicts and style bleeding
- **Iframe embedding** provides complete isolation but introduces communication overhead and performance costs
- **Different websites** have varying levels of modern browser support and security policies

## Our Solution: Progressive Enhancement

### Primary: Web Components with Shadow DOM

For modern browsers (95%+ of traffic), we use **Custom Elements with Shadow DOM**:

```javascript
class AIChatbotWidget extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    // Widget renders in isolated Shadow DOM
  }
}
```

**Benefits:**
- ✅ **Complete CSS isolation** - Host page styles cannot interfere with widget
- ✅ **Native browser performance** - No iframe overhead or communication barriers  
- ✅ **Direct API access** - Seamless communication with our backend services
- ✅ **Better mobile UX** - No nested scrolling or viewport issues
- ✅ **SEO friendly** - Content remains part of the main document

### Fallback: Iframe Embedding

For older browsers or restrictive CSP environments:

```javascript
renderIframe(shadow) {
  const iframe = document.createElement('iframe');
  iframe.src = `${this.config.widgetUrl}?orgId=${this.config.orgId}`;
  shadow.appendChild(iframe);
}
```

**Ensures:**
- ✅ **Universal compatibility** - Works in any browser that supports iframes (99.9%+)
- ✅ **Security compliance** - Meets strictest CSP and security requirements
- ✅ **Graceful degradation** - Automatic fallback with identical functionality

## Implementation Strategy

### 1. Feature Detection
```javascript
supportsWebComponents() {
  return 'customElements' in window && 
         'attachShadow' in Element.prototype &&
         'getRootNode' in Element.prototype;
}
```

### 2. Unified Configuration
Both rendering modes use identical configuration and connect to the same backend:

```javascript
window.ChatbotConfig = {
  orgId: 'customer-123',
  apiEndpoint: '/api/chat',
  theme: 'corporate-blue',
  // ... same config for both modes
};
```

### 3. Seamless Backend Integration
Whether rendered as Web Component or iframe, the widget connects to our unified:
- **RAG pipeline** for organization-specific knowledge
- **MCP servers** for customer integrations  
- **LLM routing** for intelligent responses

## Why This Architecture?

### For Customers
- **Easy integration** - Single script tag works everywhere
- **No conflicts** - Widget appearance is guaranteed consistent
- **Future-proof** - Automatically gets performance improvements as browsers update

### For Development  
- **Single codebase** - One widget implementation, multiple rendering strategies
- **Progressive enhancement** - Better experience for modern browsers, fallback for older ones
- **Simplified testing** - Same functionality across all deployment modes

### For Performance
- **Optimal by default** - Modern browsers get Web Component performance
- **No compromise fallback** - Older browsers still get full functionality
- **Minimal overhead** - Feature detection happens once during initialization

## Browser Support Matrix

| Browser | Web Components | Iframe Fallback |
|---------|----------------|-----------------|
| Chrome 67+ | ✅ Primary | ⚪ Not needed |
| Firefox 63+ | ✅ Primary | ⚪ Not needed |  
| Safari 12+ | ✅ Primary | ⚪ Not needed |
| Edge 79+ | ✅ Primary | ⚪ Not needed |
| IE 11 | ❌ Not supported | ✅ Fallback |
| Legacy browsers | ❌ Not supported | ✅ Fallback |

## Integration Examples

### CDN Integration (Recommended)
```html
<script>
  window.ChatbotConfig = { orgId: 'your-org' };
</script>
<script src="https://cdn.yourcompany.com/ai-chatbot.js"></script>
```

### NPM Integration
```javascript
import { AIChatbot } from '@yourcompany/ai-chatbot';
new AIChatbot({ orgId: 'your-org' }).init();
```

### Direct Iframe (High Security Environments)
```html
<iframe src="https://widget.yourcompany.com/embed?orgId=your-org"></iframe>
```

All three methods provide identical functionality - the widget automatically selects the optimal rendering strategy based on browser capabilities and environment constraints.

## Technical Benefits

This architecture allows us to:
- Deliver cutting-edge performance for modern users
- Maintain backward compatibility for legacy environments  
- Use a single, maintainable codebase
- Provide consistent functionality across all deployment scenarios
- Future-proof the widget as web standards evolve

The result is a robust, performant widget that works reliably across the diverse landscape of customer websites while maintaining our unified backend architecture.


# Project Structure and Files

This section explains the structure of the project and the purpose for the dirs and files.

## **Root Level Files**
```
chatbot-integration-demo/
├── README.md
├── package.json
├── .env.example
├── .gitignore
```

## **Test Server Files**
```
test-server/
├── app.js                          # Main Express application
├── views/                          # EJS templates
│   ├── index.ejs                   # CDN integration test
│   ├── npm-integration.ejs         # NPM package test
│   ├── iframe-integration.ejs      # Iframe embedding test
│   ├── mobile-test.ejs            # Mobile responsive test
│   ├── corporate-sim.ejs          # Corporate website simulation
│   ├── ecommerce-sim.ejs          # E-commerce integration
│   └── error.ejs                  # Error page template
├── public/
│   ├── css/
│   │   ├── themes/
│   │   │   ├── corporate-blue.css  # Professional blue theme
│   │   │   ├── warm-orange.css     # Friendly orange theme
│   │   │   └── modern-purple.css   # Contemporary purple theme
│   │   └── demo-sites.css          # Styling for demo pages
│   ├── js/
│   │   ├── chatbot-widget.js       # Main widget script
│   │   └── integration-examples.js # Integration helpers
│   └── images/                     # Demo logos and assets
│       └── logos/
│           ├── acme-corp.png
│           └── techshop.png
```

## **Examples Files**
```
examples/
├── vanilla-html/
│   ├── index.html                  # Pure HTML/JS integration
│   ├── style.css
│   └── script.js
├── react-app/
│   ├── package.json
│   ├── src/
│   │   ├── App.js                  # React component integration
│   │   ├── ChatbotWidget.js
│   │   └── index.js
│   └── public/
│       └── index.html
├── wordpress-plugin/
│   ├── chatbot-widget.php          # Main plugin file
│   ├── admin/
│   │   └── admin-panel.php         # WordPress admin configuration
│   └── assets/
│       ├── admin.css
│       └── admin.js
├── shopify-theme/
│   ├── layout/
│   │   └── theme.liquid            # Shopify theme integration
│   ├── snippets/
│   │   └── chatbot-widget.liquid
│   └── assets/
│       └── chatbot-config.js
└── static-site/
    ├── _config.yml                 # Jekyll configuration
    ├── _includes/
    │   └── chatbot-widget.html     # Jekyll/Hugo integration
    └── assets/
        └── chatbot-init.js
```

## **Mock Data Files**
```
mock-data/
├── organizations/
│   ├── acme-corp.json              # Sample organization config
│   ├── techshop.json
│   └── demo-org.json
├── responses/
│   ├── general-responses.json      # General conversational responses
│   ├── product-responses.json      # Product-specific responses
│   ├── error-responses.json        # Error scenario responses
│   └── contextual-responses.json   # Page-specific responses
└── sample-content/
    ├── company-info.json           # Sample company information
    ├── product-catalog.json        # Sample product data
    └── faq-content.json            # Sample FAQ content
```

## **Documentation Files**
```
documentation/
├── integration-guide.md           # Step-by-step integration instructions
├── configuration-reference.md     # Configuration options reference
├── troubleshooting.md             # Common issues and solutions
├── api-reference.md               # Mock API documentation
└── best-practices.md              # Integration best practices
```

## **Scripts Files**
```
scripts/
├── setup-mock-data.js             # Initialize mock data
├── build-examples.js              # Build example projects
└── validate-config.js             # Validate configuration files
```

## **Additional Development Files**
```
tests/
├── unit/
│   ├── widget.test.js              # Widget functionality tests
│   └── api.test.js                 # API endpoint tests
├── integration/
│   ├── server.test.js              # Server integration tests
│   └── examples.test.js            # Example integration tests
└── e2e/
    ├── widget-interactions.test.js # End-to-end widget tests
    └── mobile-responsive.test.js   # Mobile testing
```

## **Configuration Files**
```
config/
├── themes.json                     # Theme configurations
├── demo-sites.json                 # Demo site configurations
└── widget-defaults.json           # Default widget settings
```

## **Priority Order for Implementation:**
1. **Core server files**: `app.js`, basic EJS templates
2. **Widget JavaScript**: `chatbot-widget.js`
3. **Basic styling**: Theme CSS files
4. **Mock data**: Organization and response JSON files
5. **Integration examples**: Starting with vanilla HTML
6. **Documentation**: Integration guides and API reference
7. **Testing files**: Unit and integration tests

This represents approximately **40-50 files** total for a complete test harness implementation.