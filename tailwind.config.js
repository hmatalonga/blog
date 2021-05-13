/*
 ** TailwindCSS Configuration File
 **
 ** Docs: https://tailwindcss.com/docs/configuration
 ** Default: https://github.com/tailwindcss/tailwindcss/blob/master/stubs/defaultConfig.stub.js
 */

const defaultTheme = require('tailwindcss/defaultTheme')

module.exports = {
  mode: 'jit',
  future: {
    purgeLayersByDefault: true,
  },
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
        special: ['Merriweather', 'serif'],
      },
    },
    typography: (theme) => ({
      default: {
        css: {
          a: {
            color: theme('colors.indigo.600'),
          },
          'pre code': {
            fontSize: theme('fontSize.sm'),
            lineHeight: theme('lineHeight.snug'),
          },
        },
      },
    }),
  },
  variants: {},
  plugins: [require('@tailwindcss/typography')],
  purge: {
    // Learn more on https://tailwindcss.com/docs/controlling-file-size/#removing-unused-css
    enabled: process.env.NODE_ENV === 'production',
    content: [
      'components/**/*.vue',
      'layouts/**/*.vue',
      'pages/**/*.vue',
      'plugins/**/*.js',
      'nuxt.config.js',
    ],
  },
}
